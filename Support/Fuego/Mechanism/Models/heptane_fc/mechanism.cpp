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
static AMREX_GPU_DEVICE_MANAGED double imw[32] = {
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
    1.0 / 14.027090,  /*CH2GSG */
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
    1.0 / 43.045610,  /*CH3CO */
    1.0 / 40.065330,  /*C3H4XA */
    1.0 / 41.073300,  /*C3H5XA */
    1.0 / 75.088040,  /*NXC3H7O2 */
    1.0 / 54.092420,  /*C4H6 */
    1.0 / 55.100390,  /*C4H7 */
    1.0 / 56.108360,  /*C4H8X1 */
    1.0 / 99.197600,  /*C7H15X2 */
    1.0 / 100.205570};  /*NXC7H16 */

/* Molecular weights */
static AMREX_GPU_DEVICE_MANAGED double molecular_weights[32] = {
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
    14.027090,  /*CH2GSG */
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
    43.045610,  /*CH3CO */
    40.065330,  /*C3H4XA */
    41.073300,  /*C3H5XA */
    75.088040,  /*NXC3H7O2 */
    54.092420,  /*C4H6 */
    55.100390,  /*C4H7 */
    56.108360,  /*C4H8X1 */
    99.197600,  /*C7H15X2 */
    100.205570};  /*NXC7H16 */

AMREX_GPU_HOST_DEVICE
void get_imw(double imw_new[]){
    for(int i = 0; i<32; ++i) imw_new[i] = imw[i];
}

/* TODO: check necessity because redundant with CKWT */
AMREX_GPU_HOST_DEVICE
void get_mw(double mw_new[]){
    for(int i = 0; i<32; ++i) mw_new[i] = molecular_weights[i];
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
    TBid[11][0] = 40; TB[11][0] = 0; // CH2CHO
    TBid[11][1] = 32; TB[11][1] = 0; // CH
    TBid[11][2] = 10; TB[11][2] = 1.8999999999999999; // CO
    TBid[11][3] = 2; TB[11][3] = 2.5; // H2
    TBid[11][4] = 46; TB[11][4] = 0; // PXC4H9
    TBid[11][5] = 12; TB[11][5] = 0; // CH2GSG
    TBid[11][6] = 28; TB[11][6] = 0; // C4H7
    TBid[11][7] = 38; TB[11][7] = 0; // HCCO
    TBid[11][8] = 50; TB[11][8] = 0; // C5H11X1
    TBid[11][9] = 5; TB[11][9] = 12; // H2O
    TBid[11][10] = 9; TB[11][10] = 3.7999999999999998; // CO2
    TBid[11][11] = 41; TB[11][11] = 0; // C2H5O
    TBid[11][12] = 43; TB[11][12] = 0; // C3H2
    TBid[11][13] = 23; TB[11][13] = 0; // CH3CO
    TBid[11][14] = 30; TB[11][14] = 0; // C7H15X2

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
    TBid[0][0] = 40; TB[0][0] = 0; // CH2CHO
    TBid[0][1] = 32; TB[0][1] = 0; // CH
    TBid[0][2] = 10; TB[0][2] = 1.8999999999999999; // CO
    TBid[0][3] = 2; TB[0][3] = 2.5; // H2
    TBid[0][4] = 46; TB[0][4] = 0; // PXC4H9
    TBid[0][5] = 12; TB[0][5] = 0; // CH2GSG
    TBid[0][6] = 28; TB[0][6] = 0; // C4H7
    TBid[0][7] = 38; TB[0][7] = 0; // HCCO
    TBid[0][8] = 50; TB[0][8] = 0; // C5H11X1
    TBid[0][9] = 5; TB[0][9] = 12; // H2O
    TBid[0][10] = 9; TB[0][10] = 3.7999999999999998; // CO2
    TBid[0][11] = 41; TB[0][11] = 0; // C2H5O
    TBid[0][12] = 43; TB[0][12] = 0; // C3H2
    TBid[0][13] = 23; TB[0][13] = 0; // CH3CO
    TBid[0][14] = 30; TB[0][14] = 0; // C7H15X2

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
    TBid[1][0] = 40; TB[1][0] = 0; // CH2CHO
    TBid[1][1] = 32; TB[1][1] = 0; // CH
    TBid[1][2] = 10; TB[1][2] = 1.8999999999999999; // CO
    TBid[1][3] = 2; TB[1][3] = 2.5; // H2
    TBid[1][4] = 46; TB[1][4] = 0; // PXC4H9
    TBid[1][5] = 12; TB[1][5] = 0; // CH2GSG
    TBid[1][6] = 28; TB[1][6] = 0; // C4H7
    TBid[1][7] = 38; TB[1][7] = 0; // HCCO
    TBid[1][8] = 50; TB[1][8] = 0; // C5H11X1
    TBid[1][9] = 5; TB[1][9] = 12; // H2O
    TBid[1][10] = 9; TB[1][10] = 3.7999999999999998; // CO2
    TBid[1][11] = 41; TB[1][11] = 0; // C2H5O
    TBid[1][12] = 43; TB[1][12] = 0; // C3H2
    TBid[1][13] = 23; TB[1][13] = 0; // CH3CO
    TBid[1][14] = 30; TB[1][14] = 0; // C7H15X2

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
    kiv[43] = {12,3,2};
    nuv[43] = {-1,-1,1};
    kiv_qss[43] = {0};
    nuv_qss[43] = {1};
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
    kiv[12] = {12};
    nuv[12] = {-1};
    kiv_qss[12] = {2};
    nuv_qss[12] = {1};
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
    TBid[12][0] = 40; TB[12][0] = 0; // CH2CHO
    TBid[12][1] = 32; TB[12][1] = 0; // CH
    TBid[12][2] = 43; TB[12][2] = 0; // C3H2
    TBid[12][3] = 46; TB[12][3] = 0; // PXC4H9
    TBid[12][4] = 12; TB[12][4] = 0; // CH2GSG
    TBid[12][5] = 28; TB[12][5] = 0; // C4H7
    TBid[12][6] = 38; TB[12][6] = 0; // HCCO
    TBid[12][7] = 50; TB[12][7] = 0; // C5H11X1
    TBid[12][8] = 23; TB[12][8] = 0; // CH3CO
    TBid[12][9] = 41; TB[12][9] = 0; // C2H5O
    TBid[12][10] = 30; TB[12][10] = 0; // C7H15X2

    // (30):  CH2 + M => CH2GSG + M
    kiv[13] = {12};
    nuv[13] = {1};
    kiv_qss[13] = {2};
    nuv_qss[13] = {-1};
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
    TBid[13][0] = 40; TB[13][0] = 0; // CH2CHO
    TBid[13][1] = 32; TB[13][1] = 0; // CH
    TBid[13][2] = 43; TB[13][2] = 0; // C3H2
    TBid[13][3] = 46; TB[13][3] = 0; // PXC4H9
    TBid[13][4] = 12; TB[13][4] = 0; // CH2GSG
    TBid[13][5] = 28; TB[13][5] = 0; // C4H7
    TBid[13][6] = 38; TB[13][6] = 0; // HCCO
    TBid[13][7] = 50; TB[13][7] = 0; // C5H11X1
    TBid[13][8] = 23; TB[13][8] = 0; // CH3CO
    TBid[13][9] = 41; TB[13][9] = 0; // C2H5O
    TBid[13][10] = 30; TB[13][10] = 0; // C7H15X2

    // (31):  CH2GSG + H2 => CH3 + H
    kiv[44] = {12,2,13,3};
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
    kiv[45] = {13,3,12,2};
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
    kiv[46] = {12,6,10,4,3};
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
    kiv[47] = {12,4,11,3};
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
    kiv[48] = {13,4,11,2};
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
    kiv[49] = {13,4,12,5};
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
    kiv[50] = {12,5,13,4};
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
    kiv[51] = {13,1,11,3};
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
    kiv[52] = {13,7,4};
    nuv[52] = {-1,-1,1};
    kiv_qss[52] = {3};
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
    kiv[53] = {13,7,14,6};
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
    kiv[2] = {4,13,15};
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
    TBid[2][0] = 40; TB[2][0] = 0; // CH2CHO
    TBid[2][1] = 32; TB[2][1] = 0; // CH
    TBid[2][2] = 10; TB[2][2] = 1.5; // CO
    TBid[2][3] = 2; TB[2][3] = 2; // H2
    TBid[2][4] = 46; TB[2][4] = 0; // PXC4H9
    TBid[2][5] = 12; TB[2][5] = 0; // CH2GSG
    TBid[2][6] = 28; TB[2][6] = 0; // C4H7
    TBid[2][7] = 38; TB[2][7] = 0; // HCCO
    TBid[2][8] = 50; TB[2][8] = 0; // C5H11X1
    TBid[2][9] = 5; TB[2][9] = 6; // H2O
    TBid[2][10] = 9; TB[2][10] = 2; // CO2
    TBid[2][11] = 14; TB[2][11] = 2; // CH4
    TBid[2][12] = 16; TB[2][12] = 3; // C2H6
    TBid[2][13] = 41; TB[2][13] = 0; // C2H5O
    TBid[2][14] = 43; TB[2][14] = 0; // C3H2
    TBid[2][15] = 23; TB[2][15] = 0; // CH3CO
    TBid[2][16] = 30; TB[2][16] = 0; // C7H15X2

    // (42):  CH3 + O2 => CH2O + OH
    kiv[54] = {13,6,11,4};
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
    kiv[3] = {13,3,14};
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
    TBid[3][0] = 40; TB[3][0] = 0; // CH2CHO
    TBid[3][1] = 32; TB[3][1] = 0; // CH
    TBid[3][2] = 10; TB[3][2] = 2; // CO
    TBid[3][3] = 2; TB[3][3] = 2; // H2
    TBid[3][4] = 46; TB[3][4] = 0; // PXC4H9
    TBid[3][5] = 12; TB[3][5] = 0; // CH2GSG
    TBid[3][6] = 28; TB[3][6] = 0; // C4H7
    TBid[3][7] = 38; TB[3][7] = 0; // HCCO
    TBid[3][8] = 50; TB[3][8] = 0; // C5H11X1
    TBid[3][9] = 5; TB[3][9] = 5; // H2O
    TBid[3][10] = 9; TB[3][10] = 3; // CO2
    TBid[3][11] = 41; TB[3][11] = 0; // C2H5O
    TBid[3][12] = 43; TB[3][12] = 0; // C3H2
    TBid[3][13] = 23; TB[3][13] = 0; // CH3CO
    TBid[3][14] = 30; TB[3][14] = 0; // C7H15X2

    // (44):  CH3 + H => CH2 + H2
    kiv[55] = {13,3,2};
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
    kiv[56] = {2,13,3};
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
    kiv[4] = {13,16};
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
    TBid[4][0] = 40; TB[4][0] = 0; // CH2CHO
    TBid[4][1] = 32; TB[4][1] = 0; // CH
    TBid[4][2] = 10; TB[4][2] = 2; // CO
    TBid[4][3] = 2; TB[4][3] = 2; // H2
    TBid[4][4] = 46; TB[4][4] = 0; // PXC4H9
    TBid[4][5] = 12; TB[4][5] = 0; // CH2GSG
    TBid[4][6] = 28; TB[4][6] = 0; // C4H7
    TBid[4][7] = 38; TB[4][7] = 0; // HCCO
    TBid[4][8] = 50; TB[4][8] = 0; // C5H11X1
    TBid[4][9] = 5; TB[4][9] = 5; // H2O
    TBid[4][10] = 9; TB[4][10] = 3; // CO2
    TBid[4][11] = 41; TB[4][11] = 0; // C2H5O
    TBid[4][12] = 43; TB[4][12] = 0; // C3H2
    TBid[4][13] = 23; TB[4][13] = 0; // CH3CO
    TBid[4][14] = 30; TB[4][14] = 0; // C7H15X2

    // (47):  2.000000 CH3 <=> H + C2H5
    kiv[57] = {13,3};
    nuv[57] = {-2.0,1};
    kiv_qss[57] = {4};
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
    kiv[58] = {13,4,5};
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
    kiv[59] = {5,13,4};
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
    kiv[60] = {14,1,13,4};
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
    kiv[61] = {14,3,13,2};
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
    kiv[62] = {13,2,14,3};
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
    kiv[63] = {14,4,13,5};
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
    kiv[64] = {13,5,14,4};
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
    kiv[5] = {10,17};
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
    TBid[5][0] = 40; TB[5][0] = 0; // CH2CHO
    TBid[5][1] = 32; TB[5][1] = 0; // CH
    TBid[5][2] = 43; TB[5][2] = 0; // C3H2
    TBid[5][3] = 46; TB[5][3] = 0; // PXC4H9
    TBid[5][4] = 12; TB[5][4] = 0; // CH2GSG
    TBid[5][5] = 28; TB[5][5] = 0; // C4H7
    TBid[5][6] = 38; TB[5][6] = 0; // HCCO
    TBid[5][7] = 50; TB[5][7] = 0; // C5H11X1
    TBid[5][8] = 23; TB[5][8] = 0; // CH3CO
    TBid[5][9] = 41; TB[5][9] = 0; // C2H5O
    TBid[5][10] = 30; TB[5][10] = 0; // C7H15X2

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
    TBid[6][0] = 40; TB[6][0] = 0; // CH2CHO
    TBid[6][1] = 32; TB[6][1] = 0; // CH
    TBid[6][2] = 10; TB[6][2] = 1.8999999999999999; // CO
    TBid[6][3] = 2; TB[6][3] = 2.5; // H2
    TBid[6][4] = 46; TB[6][4] = 0; // PXC4H9
    TBid[6][5] = 12; TB[6][5] = 0; // CH2GSG
    TBid[6][6] = 28; TB[6][6] = 0; // C4H7
    TBid[6][7] = 38; TB[6][7] = 0; // HCCO
    TBid[6][8] = 50; TB[6][8] = 0; // C5H11X1
    TBid[6][9] = 5; TB[6][9] = 12; // H2O
    TBid[6][10] = 9; TB[6][10] = 3.7999999999999998; // CO2
    TBid[6][11] = 41; TB[6][11] = 0; // C2H5O
    TBid[6][12] = 43; TB[6][12] = 0; // C3H2
    TBid[6][13] = 23; TB[6][13] = 0; // CH3CO
    TBid[6][14] = 30; TB[6][14] = 0; // C7H15X2

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
    TBid[14][0] = 40; TB[14][0] = 0; // CH2CHO
    TBid[14][1] = 32; TB[14][1] = 0; // CH
    TBid[14][2] = 10; TB[14][2] = 1.8999999999999999; // CO
    TBid[14][3] = 2; TB[14][3] = 2.5; // H2
    TBid[14][4] = 46; TB[14][4] = 0; // PXC4H9
    TBid[14][5] = 12; TB[14][5] = 0; // CH2GSG
    TBid[14][6] = 28; TB[14][6] = 0; // C4H7
    TBid[14][7] = 38; TB[14][7] = 0; // HCCO
    TBid[14][8] = 50; TB[14][8] = 0; // C5H11X1
    TBid[14][9] = 5; TB[14][9] = 6; // H2O
    TBid[14][10] = 9; TB[14][10] = 3.7999999999999998; // CO2
    TBid[14][11] = 41; TB[14][11] = 0; // C2H5O
    TBid[14][12] = 43; TB[14][12] = 0; // C3H2
    TBid[14][13] = 23; TB[14][13] = 0; // CH3CO
    TBid[14][14] = 30; TB[14][14] = 0; // C7H15X2

    // (65):  HCO + CH3 => CH4 + CO
    kiv[72] = {13,14,10};
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
    kiv[76] = {11,13,14};
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
    kiv[77] = {15,11};
    nuv[77] = {1,1};
    kiv_qss[77] = {3};
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
    kiv_qss[78] = {3};
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
    kiv_qss[7] = {3};
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
    TBid[7][0] = 40; TB[7][0] = 0; // CH2CHO
    TBid[7][1] = 32; TB[7][1] = 0; // CH
    TBid[7][2] = 43; TB[7][2] = 0; // C3H2
    TBid[7][3] = 46; TB[7][3] = 0; // PXC4H9
    TBid[7][4] = 12; TB[7][4] = 0; // CH2GSG
    TBid[7][5] = 28; TB[7][5] = 0; // C4H7
    TBid[7][6] = 38; TB[7][6] = 0; // HCCO
    TBid[7][7] = 50; TB[7][7] = 0; // C5H11X1
    TBid[7][8] = 23; TB[7][8] = 0; // CH3CO
    TBid[7][9] = 41; TB[7][9] = 0; // C2H5O
    TBid[7][10] = 30; TB[7][10] = 0; // C7H15X2

    // (73):  CH3O + H2 => CH3OH + H
    kiv[79] = {2,15,3};
    nuv[79] = {-1,1,1};
    kiv_qss[79] = {3};
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
    kiv[80] = {15,4,5};
    nuv[80] = {-1,-1,1};
    kiv_qss[80] = {3};
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
    kiv[81] = {12,9,11,10};
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
    kiv[82] = {18,3,2,10,4};
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
    kiv[83] = {18,4,5,10,4};
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
    kiv[84] = {18,4};
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
    kiv[85] = {4,18};
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
    kiv[86] = {18,3,2,9,3};
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
    kiv[87] = {18,4,5,9,3};
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
    kiv[88] = {19,6};
    nuv[88] = {-2.0,1};
    kiv_qss[88] = {3};
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
    kiv[89] = {19,13};
    nuv[89] = {-1,-1};
    kiv_qss[89] = {3};
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
    kiv[90] = {19,11,15,6};
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
    kiv[91] = {19,7,6};
    nuv[91] = {-1,-1,1};
    kiv_qss[91] = {5};
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
    kiv[15] = {19,13,6};
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
    TBid[15][0] = 40; TB[15][0] = 0; // CH2CHO
    TBid[15][1] = 32; TB[15][1] = 0; // CH
    TBid[15][2] = 43; TB[15][2] = 0; // C3H2
    TBid[15][3] = 46; TB[15][3] = 0; // PXC4H9
    TBid[15][4] = 12; TB[15][4] = 0; // CH2GSG
    TBid[15][5] = 28; TB[15][5] = 0; // C4H7
    TBid[15][6] = 38; TB[15][6] = 0; // HCCO
    TBid[15][7] = 50; TB[15][7] = 0; // C5H11X1
    TBid[15][8] = 23; TB[15][8] = 0; // CH3CO
    TBid[15][9] = 41; TB[15][9] = 0; // C2H5O
    TBid[15][10] = 30; TB[15][10] = 0; // C7H15X2

    // (87):  CH3 + O2 + M => CH3O2 + M
    kiv[16] = {13,6,19};
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
    TBid[16][0] = 40; TB[16][0] = 0; // CH2CHO
    TBid[16][1] = 32; TB[16][1] = 0; // CH
    TBid[16][2] = 43; TB[16][2] = 0; // C3H2
    TBid[16][3] = 46; TB[16][3] = 0; // PXC4H9
    TBid[16][4] = 12; TB[16][4] = 0; // CH2GSG
    TBid[16][5] = 28; TB[16][5] = 0; // C4H7
    TBid[16][6] = 38; TB[16][6] = 0; // HCCO
    TBid[16][7] = 50; TB[16][7] = 0; // C5H11X1
    TBid[16][8] = 23; TB[16][8] = 0; // CH3CO
    TBid[16][9] = 41; TB[16][9] = 0; // C2H5O
    TBid[16][10] = 30; TB[16][10] = 0; // C7H15X2

    // (88):  CH3O2H => CH3O + OH
    kiv[92] = {4};
    nuv[92] = {1};
    kiv_qss[92] = {5,3};
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
    kiv[93] = {20,1,10};
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
    kiv[94] = {20,1,3};
    nuv[94] = {-1,-1,1};
    kiv_qss[94] = {6};
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
    kiv[95] = {3,20,2};
    nuv[95] = {-1,1,1};
    kiv_qss[95] = {7};
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
    kiv_qss[96] = {7,8};
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
    kiv[97] = {13,21};
    nuv[97] = {-1,1};
    kiv_qss[97] = {7};
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
    kiv[98] = {6,20,7};
    nuv[98] = {-1,1,1};
    kiv_qss[98] = {7};
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
    kiv_qss[99] = {7,1};
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
    kiv[8] = {3,20};
    nuv[8] = {1,1};
    kiv_qss[8] = {7};
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
    TBid[8][0] = 40; TB[8][0] = 0; // CH2CHO
    TBid[8][1] = 32; TB[8][1] = 0; // CH
    TBid[8][2] = 10; TB[8][2] = 2; // CO
    TBid[8][3] = 2; TB[8][3] = 2; // H2
    TBid[8][4] = 46; TB[8][4] = 0; // PXC4H9
    TBid[8][5] = 12; TB[8][5] = 0; // CH2GSG
    TBid[8][6] = 28; TB[8][6] = 0; // C4H7
    TBid[8][7] = 38; TB[8][7] = 0; // HCCO
    TBid[8][8] = 50; TB[8][8] = 0; // C5H11X1
    TBid[8][9] = 5; TB[8][9] = 5; // H2O
    TBid[8][10] = 9; TB[8][10] = 3; // CO2
    TBid[8][11] = 41; TB[8][11] = 0; // C2H5O
    TBid[8][12] = 43; TB[8][12] = 0; // C3H2
    TBid[8][13] = 23; TB[8][13] = 0; // CH3CO
    TBid[8][14] = 30; TB[8][14] = 0; // C7H15X2

    // (97):  C2H4 + CH3 => C2H3 + CH4
    kiv[100] = {22,13,14};
    nuv[100] = {-1,-1,1};
    kiv_qss[100] = {7};
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
    kiv[101] = {22,1,13};
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
    kiv[102] = {22,4,5};
    nuv[102] = {-1,-1,1};
    kiv_qss[102] = {7};
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
    kiv[9] = {3,22};
    nuv[9] = {-1,-1};
    kiv_qss[9] = {4};
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
    TBid[9][0] = 40; TB[9][0] = 0; // CH2CHO
    TBid[9][1] = 32; TB[9][1] = 0; // CH
    TBid[9][2] = 43; TB[9][2] = 0; // C3H2
    TBid[9][3] = 46; TB[9][3] = 0; // PXC4H9
    TBid[9][4] = 12; TB[9][4] = 0; // CH2GSG
    TBid[9][5] = 28; TB[9][5] = 0; // C4H7
    TBid[9][6] = 38; TB[9][6] = 0; // HCCO
    TBid[9][7] = 50; TB[9][7] = 0; // C5H11X1
    TBid[9][8] = 23; TB[9][8] = 0; // CH3CO
    TBid[9][9] = 41; TB[9][9] = 0; // C2H5O
    TBid[9][10] = 30; TB[9][10] = 0; // C7H15X2

    // (101):  C2H4 + O => CH2CHO + H
    kiv[103] = {22,1,3};
    nuv[103] = {-1,-1,1};
    kiv_qss[103] = {8};
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
    kiv[104] = {22,3,2};
    nuv[104] = {-1,-1,1};
    kiv_qss[104] = {7};
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
    kiv[105] = {2,22,3};
    nuv[105] = {-1,1,1};
    kiv_qss[105] = {7};
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
    kiv[106] = {3,16};
    nuv[106] = {-1,1};
    kiv_qss[106] = {4};
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
    kiv[107] = {19};
    nuv[107] = {-1};
    kiv_qss[107] = {4,3,9};
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
    kiv_qss[108] = {4,9};
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
    kiv[109] = {6,22,7};
    nuv[109] = {-1,1,1};
    kiv_qss[109] = {4};
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
    kiv[110] = {16,1,4};
    nuv[110] = {-1,-1,1};
    kiv_qss[110] = {4};
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
    kiv[111] = {16,4,5};
    nuv[111] = {-1,-1,1};
    kiv_qss[111] = {4};
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
    kiv[112] = {16,3,2};
    nuv[112] = {-1,-1,1};
    kiv_qss[112] = {4};
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
    kiv_qss[113] = {6};
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
    kiv_qss[114] = {6,1};
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
    kiv_qss[115] = {6,1};
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
    kiv[116] = {3,12,10};
    nuv[116] = {-1,1,1};
    kiv_qss[116] = {6};
    nuv_qss[116] = {-1};
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
    kiv[117] = {12,10,3};
    nuv[117] = {-1,-1,1};
    kiv_qss[117] = {6};
    nuv_qss[117] = {1};
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
    kiv[118] = {17,1,4};
    nuv[118] = {-1,-1,1};
    kiv_qss[118] = {6};
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
    kiv[119] = {17,3,2};
    nuv[119] = {-1,-1,1};
    kiv_qss[119] = {6};
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
    kiv[120] = {2,17,3};
    nuv[120] = {-1,1,1};
    kiv_qss[120] = {6};
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
    kiv[121] = {17,3,13,10};
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
    kiv[122] = {17,1,9};
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
    kiv[123] = {17,4,5};
    nuv[123] = {-1,-1,1};
    kiv_qss[123] = {6};
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
    kiv_qss[124] = {8};
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
    kiv[125] = {17,3};
    nuv[125] = {1,1};
    kiv_qss[125] = {8};
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
    kiv[126] = {17,3};
    nuv[126] = {-1,-1};
    kiv_qss[126] = {8};
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
    kiv[10] = {23,13,10};
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
    TBid[10][0] = 40; TB[10][0] = 0; // CH2CHO
    TBid[10][1] = 32; TB[10][1] = 0; // CH
    TBid[10][2] = 43; TB[10][2] = 0; // C3H2
    TBid[10][3] = 46; TB[10][3] = 0; // PXC4H9
    TBid[10][4] = 12; TB[10][4] = 0; // CH2GSG
    TBid[10][5] = 28; TB[10][5] = 0; // C4H7
    TBid[10][6] = 38; TB[10][6] = 0; // HCCO
    TBid[10][7] = 50; TB[10][7] = 0; // C5H11X1
    TBid[10][8] = 23; TB[10][8] = 0; // CH3CO
    TBid[10][9] = 41; TB[10][9] = 0; // C2H5O
    TBid[10][10] = 30; TB[10][10] = 0; // C7H15X2

    // (126):  C2H5O + M => CH3 + CH2O + M
    kiv[17] = {13,11};
    nuv[17] = {1,1};
    kiv_qss[17] = {9};
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
    TBid[17][0] = 40; TB[17][0] = 0; // CH2CHO
    TBid[17][1] = 32; TB[17][1] = 0; // CH
    TBid[17][2] = 43; TB[17][2] = 0; // C3H2
    TBid[17][3] = 46; TB[17][3] = 0; // PXC4H9
    TBid[17][4] = 12; TB[17][4] = 0; // CH2GSG
    TBid[17][5] = 28; TB[17][5] = 0; // C4H7
    TBid[17][6] = 38; TB[17][6] = 0; // HCCO
    TBid[17][7] = 50; TB[17][7] = 0; // C5H11X1
    TBid[17][8] = 23; TB[17][8] = 0; // CH3CO
    TBid[17][9] = 41; TB[17][9] = 0; // C2H5O
    TBid[17][10] = 30; TB[17][10] = 0; // C7H15X2

    // (127):  C2H5O2 => C2H5 + O2
    kiv[127] = {6};
    nuv[127] = {1};
    kiv_qss[127] = {10,4};
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
    kiv_qss[128] = {4,10};
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
    kiv[129] = {22,7};
    nuv[129] = {1,1};
    kiv_qss[129] = {10};
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
    kiv_qss[130] = {11,6};
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
    kiv[131] = {4,20};
    nuv[131] = {-1,1};
    kiv_qss[131] = {11,1};
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
    kiv[132] = {6,17};
    nuv[132] = {-1,1};
    kiv_qss[132] = {12,1};
    nuv_qss[132] = {-1,1};
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
    kiv[133] = {7,24,6};
    nuv[133] = {-1,1,1};
    kiv_qss[133] = {12};
    nuv_qss[133] = {-1};
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
    kiv[134] = {3,2};
    nuv[134] = {-1,1};
    kiv_qss[134] = {12,11};
    nuv_qss[134] = {-1,1};
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
    kiv[135] = {4,5};
    nuv[135] = {-1,1};
    kiv_qss[135] = {12,11};
    nuv_qss[135] = {-1,1};
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
    kiv[136] = {5,4};
    nuv[136] = {-1,1};
    kiv_qss[136] = {11,12};
    nuv_qss[136] = {-1,1};
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
    kiv[137] = {24,3,2};
    nuv[137] = {-1,-1,1};
    kiv_qss[137] = {12};
    nuv_qss[137] = {1};
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
    kiv[138] = {24,4,5};
    nuv[138] = {-1,-1,1};
    kiv_qss[138] = {12};
    nuv_qss[138] = {1};
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
    kiv[139] = {24,1,22,10};
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
    kiv[140] = {25,3,24,2};
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
    kiv[141] = {25,7,21,6};
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
    kiv[142] = {25,3,21};
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
    kiv[143] = {25,20,13};
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
    kiv[144] = {25,24,3};
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
    kiv[145] = {24,3,25};
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
    kiv[146] = {25,11,21};
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
    kiv[147] = {25,24,21};
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
    kiv[148] = {21,3,22,13};
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
    kiv[149] = {21,3,25,2};
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
    kiv[150] = {21,1};
    nuv[150] = {-1,-1};
    kiv_qss[150] = {4,1};
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
    kiv[151] = {21,1,25,4};
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
    kiv[152] = {21,1,17,13,3};
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
    kiv[153] = {21,4,25,5};
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
    kiv[154] = {6,21,7};
    nuv[154] = {-1,1,1};
    kiv_qss[154] = {13};
    nuv_qss[154] = {-1};
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
    kiv[155] = {13,22};
    nuv[155] = {1,1};
    kiv_qss[155] = {13};
    nuv_qss[155] = {-1};
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
    kiv[156] = {13,22};
    nuv[156] = {-1,-1};
    kiv_qss[156] = {13};
    nuv_qss[156] = {1};
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
    kiv[157] = {3,21};
    nuv[157] = {1,1};
    kiv_qss[157] = {13};
    nuv_qss[157] = {-1};
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
    kiv[158] = {3,21};
    nuv[158] = {-1,-1};
    kiv_qss[158] = {13};
    nuv_qss[158] = {1};
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
    kiv[159] = {26,6};
    nuv[159] = {-1,1};
    kiv_qss[159] = {13};
    nuv_qss[159] = {1};
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
    kiv[160] = {6,26};
    nuv[160] = {-1,1};
    kiv_qss[160] = {13};
    nuv_qss[160] = {-1};
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
    kiv_qss[161] = {7};
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
    kiv_qss[162] = {7};
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
    kiv[163] = {27,4,11,25};
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
    kiv[164] = {27,4,17};
    nuv[164] = {-1,-1,1};
    kiv_qss[164] = {4};
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
    kiv[165] = {27,1,22,17};
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
    kiv[166] = {27,3,22};
    nuv[166] = {-1,-1,1};
    kiv_qss[166] = {7};
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
    kiv[167] = {27,1,11,24};
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
    kiv[169] = {25,28,21,27};
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
    kiv[170] = {28,27,16};
    nuv[170] = {-1,1,1};
    kiv_qss[170] = {4};
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
    kiv[173] = {28,13,27,14};
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
    kiv[176] = {28,22};
    nuv[176] = {-1,1};
    kiv_qss[176] = {7};
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
    kiv[179] = {29,4,11};
    nuv[179] = {-1,-1,1};
    kiv_qss[179] = {13};
    nuv_qss[179] = {1};
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
    kiv[180] = {29,4,23,16};
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
    kiv[181] = {29,1,23};
    nuv[181] = {-1,-1,1};
    kiv_qss[181] = {4};
    nuv_qss[181] = {1};
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
    kiv[182] = {29,1,21,11};
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
    kiv[184] = {29,25,13};
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
    kiv[185] = {25,13,29};
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
    kiv[186] = {29,3};
    nuv[186] = {1,1};
    kiv_qss[186] = {14};
    nuv_qss[186] = {-1};
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
    kiv[187] = {29,3};
    nuv[187] = {-1,-1};
    kiv_qss[187] = {14};
    nuv_qss[187] = {1};
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
    kiv[188] = {22};
    nuv[188] = {1};
    kiv_qss[188] = {14,4};
    nuv_qss[188] = {-1,1};
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
    kiv[189] = {6};
    nuv[189] = {1};
    kiv_qss[189] = {15,14};
    nuv_qss[189] = {-1,1};
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
    kiv[190] = {6};
    nuv[190] = {-1};
    kiv_qss[190] = {14,15};
    nuv_qss[190] = {-1,1};
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
    kiv[191] = {27,13};
    nuv[191] = {1,1};
    kiv_qss[191] = {16};
    nuv_qss[191] = {-1};
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
    kiv[192] = {25,22};
    nuv[192] = {1,1};
    kiv_qss[192] = {16};
    nuv_qss[192] = {-1};
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
    kiv[193] = {4,5};
    nuv[193] = {-1,1};
    kiv_qss[193] = {17,16};
    nuv_qss[193] = {-1,1};
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
    kiv[194] = {3,2};
    nuv[194] = {-1,1};
    kiv_qss[194] = {17,16};
    nuv_qss[194] = {-1,1};
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
    kiv[195] = {25};
    nuv[195] = {1};
    kiv_qss[195] = {17,4};
    nuv_qss[195] = {-1,1};
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
    kiv[196] = {25};
    nuv[196] = {-1};
    kiv_qss[196] = {4,17};
    nuv_qss[196] = {-1,1};
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
    kiv[197] = {1,4};
    nuv[197] = {-1,1};
    kiv_qss[197] = {17,16};
    nuv_qss[197] = {-1,1};
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
    kiv[198] = {21};
    nuv[198] = {1};
    kiv_qss[198] = {18,4};
    nuv_qss[198] = {-1,1};
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
    kiv[199] = {22};
    nuv[199] = {1};
    kiv_qss[199] = {18,13};
    nuv_qss[199] = {-1,1};
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
    kiv[200] = {3};
    nuv[200] = {1};
    kiv_qss[200] = {18,17};
    nuv_qss[200] = {-1,1};
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
    kiv[201] = {25};
    nuv[201] = {1};
    kiv_qss[201] = {19,13};
    nuv_qss[201] = {-1,1};
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
    kiv[202] = {4,11};
    nuv[202] = {-1,1};
    kiv_qss[202] = {19,18};
    nuv_qss[202] = {-1,1};
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
    kiv[203] = {30,13};
    nuv[203] = {-1,1};
    kiv_qss[203] = {19};
    nuv_qss[203] = {1};
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
    kiv[204] = {30,21};
    nuv[204] = {-1,1};
    kiv_qss[204] = {14};
    nuv_qss[204] = {1};
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
    kiv[205] = {30,29};
    nuv[205] = {-1,1};
    kiv_qss[205] = {13};
    nuv_qss[205] = {1};
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
    kiv[206] = {30,22};
    nuv[206] = {-1,1};
    kiv_qss[206] = {18};
    nuv_qss[206] = {1};
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
    kiv[207] = {30};
    nuv[207] = {-1};
    kiv_qss[207] = {4,17};
    nuv_qss[207] = {1,1};
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
    kiv[208] = {30,7,31,6};
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
    kiv[209] = {31,19,30};
    nuv[209] = {-1,-1,1};
    kiv_qss[209] = {5};
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
    kiv[210] = {31,3,30,2};
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
    kiv[211] = {31};
    nuv[211] = {-1};
    kiv_qss[211] = {14,13};
    nuv_qss[211] = {1,1};
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
    kiv[212] = {31,7,30,8};
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
    kiv[213] = {31};
    nuv[213] = {-1};
    kiv_qss[213] = {18,4};
    nuv_qss[213] = {1,1};
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
    kiv[214] = {31,30,15};
    nuv[214] = {-1,1,1};
    kiv_qss[214] = {3};
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
    kiv[215] = {31,1,30,4};
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
    kiv[216] = {31,4,30,5};
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
    kiv[217] = {31,13,30,14};
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
    *kk = 32;
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
    kname.resize(32);
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
    kname[12] = "CH2GSG";
    kname[13] = "CH3";
    kname[14] = "CH4";
    kname[15] = "CH3OH";
    kname[16] = "C2H6";
    kname[17] = "CH2CO";
    kname[18] = "HOCHO";
    kname[19] = "CH3O2";
    kname[20] = "C2H2";
    kname[21] = "C3H6";
    kname[22] = "C2H4";
    kname[23] = "CH3CO";
    kname[24] = "C3H4XA";
    kname[25] = "C3H5XA";
    kname[26] = "NXC3H7O2";
    kname[27] = "C4H6";
    kname[28] = "C4H7";
    kname[29] = "C4H8X1";
    kname[30] = "C7H15X2";
    kname[31] = "NXC7H16";
}


/* Returns the char strings of species names */
void CKSYMS(int * kname, int * plenkname )
{
    int i; /*Loop Counter */
    int lenkname = *plenkname;
    /*clear kname */
    for (i=0; i<lenkname*32; i++) {
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

    /* CH2GSG  */
    kname[ 12*lenkname + 0 ] = 'C';
    kname[ 12*lenkname + 1 ] = 'H';
    kname[ 12*lenkname + 2 ] = '2';
    kname[ 12*lenkname + 3 ] = 'G';
    kname[ 12*lenkname + 4 ] = 'S';
    kname[ 12*lenkname + 5 ] = 'G';
    kname[ 12*lenkname + 6 ] = ' ';

    /* CH3  */
    kname[ 13*lenkname + 0 ] = 'C';
    kname[ 13*lenkname + 1 ] = 'H';
    kname[ 13*lenkname + 2 ] = '3';
    kname[ 13*lenkname + 3 ] = ' ';

    /* CH4  */
    kname[ 14*lenkname + 0 ] = 'C';
    kname[ 14*lenkname + 1 ] = 'H';
    kname[ 14*lenkname + 2 ] = '4';
    kname[ 14*lenkname + 3 ] = ' ';

    /* CH3OH  */
    kname[ 15*lenkname + 0 ] = 'C';
    kname[ 15*lenkname + 1 ] = 'H';
    kname[ 15*lenkname + 2 ] = '3';
    kname[ 15*lenkname + 3 ] = 'O';
    kname[ 15*lenkname + 4 ] = 'H';
    kname[ 15*lenkname + 5 ] = ' ';

    /* C2H6  */
    kname[ 16*lenkname + 0 ] = 'C';
    kname[ 16*lenkname + 1 ] = '2';
    kname[ 16*lenkname + 2 ] = 'H';
    kname[ 16*lenkname + 3 ] = '6';
    kname[ 16*lenkname + 4 ] = ' ';

    /* CH2CO  */
    kname[ 17*lenkname + 0 ] = 'C';
    kname[ 17*lenkname + 1 ] = 'H';
    kname[ 17*lenkname + 2 ] = '2';
    kname[ 17*lenkname + 3 ] = 'C';
    kname[ 17*lenkname + 4 ] = 'O';
    kname[ 17*lenkname + 5 ] = ' ';

    /* HOCHO  */
    kname[ 18*lenkname + 0 ] = 'H';
    kname[ 18*lenkname + 1 ] = 'O';
    kname[ 18*lenkname + 2 ] = 'C';
    kname[ 18*lenkname + 3 ] = 'H';
    kname[ 18*lenkname + 4 ] = 'O';
    kname[ 18*lenkname + 5 ] = ' ';

    /* CH3O2  */
    kname[ 19*lenkname + 0 ] = 'C';
    kname[ 19*lenkname + 1 ] = 'H';
    kname[ 19*lenkname + 2 ] = '3';
    kname[ 19*lenkname + 3 ] = 'O';
    kname[ 19*lenkname + 4 ] = '2';
    kname[ 19*lenkname + 5 ] = ' ';

    /* C2H2  */
    kname[ 20*lenkname + 0 ] = 'C';
    kname[ 20*lenkname + 1 ] = '2';
    kname[ 20*lenkname + 2 ] = 'H';
    kname[ 20*lenkname + 3 ] = '2';
    kname[ 20*lenkname + 4 ] = ' ';

    /* C3H6  */
    kname[ 21*lenkname + 0 ] = 'C';
    kname[ 21*lenkname + 1 ] = '3';
    kname[ 21*lenkname + 2 ] = 'H';
    kname[ 21*lenkname + 3 ] = '6';
    kname[ 21*lenkname + 4 ] = ' ';

    /* C2H4  */
    kname[ 22*lenkname + 0 ] = 'C';
    kname[ 22*lenkname + 1 ] = '2';
    kname[ 22*lenkname + 2 ] = 'H';
    kname[ 22*lenkname + 3 ] = '4';
    kname[ 22*lenkname + 4 ] = ' ';

    /* CH3CO  */
    kname[ 23*lenkname + 0 ] = 'C';
    kname[ 23*lenkname + 1 ] = 'H';
    kname[ 23*lenkname + 2 ] = '3';
    kname[ 23*lenkname + 3 ] = 'C';
    kname[ 23*lenkname + 4 ] = 'O';
    kname[ 23*lenkname + 5 ] = ' ';

    /* C3H4XA  */
    kname[ 24*lenkname + 0 ] = 'C';
    kname[ 24*lenkname + 1 ] = '3';
    kname[ 24*lenkname + 2 ] = 'H';
    kname[ 24*lenkname + 3 ] = '4';
    kname[ 24*lenkname + 4 ] = 'X';
    kname[ 24*lenkname + 5 ] = 'A';
    kname[ 24*lenkname + 6 ] = ' ';

    /* C3H5XA  */
    kname[ 25*lenkname + 0 ] = 'C';
    kname[ 25*lenkname + 1 ] = '3';
    kname[ 25*lenkname + 2 ] = 'H';
    kname[ 25*lenkname + 3 ] = '5';
    kname[ 25*lenkname + 4 ] = 'X';
    kname[ 25*lenkname + 5 ] = 'A';
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

    /* C7H15X2  */
    kname[ 30*lenkname + 0 ] = 'C';
    kname[ 30*lenkname + 1 ] = '7';
    kname[ 30*lenkname + 2 ] = 'H';
    kname[ 30*lenkname + 3 ] = '1';
    kname[ 30*lenkname + 4 ] = '5';
    kname[ 30*lenkname + 5 ] = 'X';
    kname[ 30*lenkname + 6 ] = '2';
    kname[ 30*lenkname + 7 ] = ' ';

    /* NXC7H16  */
    kname[ 31*lenkname + 0 ] = 'N';
    kname[ 31*lenkname + 1 ] = 'X';
    kname[ 31*lenkname + 2 ] = 'C';
    kname[ 31*lenkname + 3 ] = '7';
    kname[ 31*lenkname + 4 ] = 'H';
    kname[ 31*lenkname + 5 ] = '1';
    kname[ 31*lenkname + 6 ] = '6';
    kname[ 31*lenkname + 7 ] = ' ';

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
    XW += x[12]*molecular_weights[12]; /*CH2GSG */
    XW += x[13]*molecular_weights[13]; /*CH3 */
    XW += x[14]*molecular_weights[14]; /*CH4 */
    XW += x[15]*molecular_weights[15]; /*CH3OH */
    XW += x[16]*molecular_weights[16]; /*C2H6 */
    XW += x[17]*molecular_weights[17]; /*CH2CO */
    XW += x[18]*molecular_weights[18]; /*HOCHO */
    XW += x[19]*molecular_weights[19]; /*CH3O2 */
    XW += x[20]*molecular_weights[20]; /*C2H2 */
    XW += x[21]*molecular_weights[21]; /*C3H6 */
    XW += x[22]*molecular_weights[22]; /*C2H4 */
    XW += x[23]*molecular_weights[23]; /*CH3CO */
    XW += x[24]*molecular_weights[24]; /*C3H4XA */
    XW += x[25]*molecular_weights[25]; /*C3H5XA */
    XW += x[26]*molecular_weights[26]; /*NXC3H7O2 */
    XW += x[27]*molecular_weights[27]; /*C4H6 */
    XW += x[28]*molecular_weights[28]; /*C4H7 */
    XW += x[29]*molecular_weights[29]; /*C4H8X1 */
    XW += x[30]*molecular_weights[30]; /*C7H15X2 */
    XW += x[31]*molecular_weights[31]; /*NXC7H16 */
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
    YOW += y[12]*imw[12]; /*CH2GSG */
    YOW += y[13]*imw[13]; /*CH3 */
    YOW += y[14]*imw[14]; /*CH4 */
    YOW += y[15]*imw[15]; /*CH3OH */
    YOW += y[16]*imw[16]; /*C2H6 */
    YOW += y[17]*imw[17]; /*CH2CO */
    YOW += y[18]*imw[18]; /*HOCHO */
    YOW += y[19]*imw[19]; /*CH3O2 */
    YOW += y[20]*imw[20]; /*C2H2 */
    YOW += y[21]*imw[21]; /*C3H6 */
    YOW += y[22]*imw[22]; /*C2H4 */
    YOW += y[23]*imw[23]; /*CH3CO */
    YOW += y[24]*imw[24]; /*C3H4XA */
    YOW += y[25]*imw[25]; /*C3H5XA */
    YOW += y[26]*imw[26]; /*NXC3H7O2 */
    YOW += y[27]*imw[27]; /*C4H6 */
    YOW += y[28]*imw[28]; /*C4H7 */
    YOW += y[29]*imw[29]; /*C4H8X1 */
    YOW += y[30]*imw[30]; /*C7H15X2 */
    YOW += y[31]*imw[31]; /*NXC7H16 */
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
    W += c[12]*14.027090; /*CH2GSG */
    W += c[13]*15.035060; /*CH3 */
    W += c[14]*16.043030; /*CH4 */
    W += c[15]*32.042430; /*CH3OH */
    W += c[16]*30.070120; /*C2H6 */
    W += c[17]*42.037640; /*CH2CO */
    W += c[18]*46.025890; /*HOCHO */
    W += c[19]*47.033860; /*CH3O2 */
    W += c[20]*26.038240; /*C2H2 */
    W += c[21]*42.081270; /*C3H6 */
    W += c[22]*28.054180; /*C2H4 */
    W += c[23]*43.045610; /*CH3CO */
    W += c[24]*40.065330; /*C3H4XA */
    W += c[25]*41.073300; /*C3H5XA */
    W += c[26]*75.088040; /*NXC3H7O2 */
    W += c[27]*54.092420; /*C4H6 */
    W += c[28]*55.100390; /*C4H7 */
    W += c[29]*56.108360; /*C4H8X1 */
    W += c[30]*99.197600; /*C7H15X2 */
    W += c[31]*100.205570; /*NXC7H16 */

    for (id = 0; id < 32; ++id) {
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
    XW += x[12]*14.027090; /*CH2GSG */
    XW += x[13]*15.035060; /*CH3 */
    XW += x[14]*16.043030; /*CH4 */
    XW += x[15]*32.042430; /*CH3OH */
    XW += x[16]*30.070120; /*C2H6 */
    XW += x[17]*42.037640; /*CH2CO */
    XW += x[18]*46.025890; /*HOCHO */
    XW += x[19]*47.033860; /*CH3O2 */
    XW += x[20]*26.038240; /*C2H2 */
    XW += x[21]*42.081270; /*C3H6 */
    XW += x[22]*28.054180; /*C2H4 */
    XW += x[23]*43.045610; /*CH3CO */
    XW += x[24]*40.065330; /*C3H4XA */
    XW += x[25]*41.073300; /*C3H5XA */
    XW += x[26]*75.088040; /*NXC3H7O2 */
    XW += x[27]*54.092420; /*C4H6 */
    XW += x[28]*55.100390; /*C4H7 */
    XW += x[29]*56.108360; /*C4H8X1 */
    XW += x[30]*99.197600; /*C7H15X2 */
    XW += x[31]*100.205570; /*NXC7H16 */
    *rho = *P * XW / (8.31446e+07 * (*T)); /*rho = P*W/(R*T) */

    return;
}


/*Compute rho = P*W(y)/RT */
AMREX_GPU_HOST_DEVICE void CKRHOY(double *  P, double *  T, double *  y,  double *  rho)
{
    double YOW = 0;
    double tmp[32];

    for (int i = 0; i < 32; i++)
    {
        tmp[i] = y[i]*imw[i];
    }
    for (int i = 0; i < 32; i++)
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
    W += c[12]*14.027090; /*CH2GSG */
    W += c[13]*15.035060; /*CH3 */
    W += c[14]*16.043030; /*CH4 */
    W += c[15]*32.042430; /*CH3OH */
    W += c[16]*30.070120; /*C2H6 */
    W += c[17]*42.037640; /*CH2CO */
    W += c[18]*46.025890; /*HOCHO */
    W += c[19]*47.033860; /*CH3O2 */
    W += c[20]*26.038240; /*C2H2 */
    W += c[21]*42.081270; /*C3H6 */
    W += c[22]*28.054180; /*C2H4 */
    W += c[23]*43.045610; /*CH3CO */
    W += c[24]*40.065330; /*C3H4XA */
    W += c[25]*41.073300; /*C3H5XA */
    W += c[26]*75.088040; /*NXC3H7O2 */
    W += c[27]*54.092420; /*C4H6 */
    W += c[28]*55.100390; /*C4H7 */
    W += c[29]*56.108360; /*C4H8X1 */
    W += c[30]*99.197600; /*C7H15X2 */
    W += c[31]*100.205570; /*NXC7H16 */

    for (id = 0; id < 32; ++id) {
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
    double tmp[32];

    for (int i = 0; i < 32; i++)
    {
        tmp[i] = y[i]*imw[i];
    }
    for (int i = 0; i < 32; i++)
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
    XW += x[12]*molecular_weights[12]; /*CH2GSG */
    XW += x[13]*molecular_weights[13]; /*CH3 */
    XW += x[14]*molecular_weights[14]; /*CH4 */
    XW += x[15]*molecular_weights[15]; /*CH3OH */
    XW += x[16]*molecular_weights[16]; /*C2H6 */
    XW += x[17]*molecular_weights[17]; /*CH2CO */
    XW += x[18]*molecular_weights[18]; /*HOCHO */
    XW += x[19]*molecular_weights[19]; /*CH3O2 */
    XW += x[20]*molecular_weights[20]; /*C2H2 */
    XW += x[21]*molecular_weights[21]; /*C3H6 */
    XW += x[22]*molecular_weights[22]; /*C2H4 */
    XW += x[23]*molecular_weights[23]; /*CH3CO */
    XW += x[24]*molecular_weights[24]; /*C3H4XA */
    XW += x[25]*molecular_weights[25]; /*C3H5XA */
    XW += x[26]*molecular_weights[26]; /*NXC3H7O2 */
    XW += x[27]*molecular_weights[27]; /*C4H6 */
    XW += x[28]*molecular_weights[28]; /*C4H7 */
    XW += x[29]*molecular_weights[29]; /*C4H8X1 */
    XW += x[30]*molecular_weights[30]; /*C7H15X2 */
    XW += x[31]*molecular_weights[31]; /*NXC7H16 */
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
    W += c[12]*molecular_weights[12]; /*CH2GSG */
    W += c[13]*molecular_weights[13]; /*CH3 */
    W += c[14]*molecular_weights[14]; /*CH4 */
    W += c[15]*molecular_weights[15]; /*CH3OH */
    W += c[16]*molecular_weights[16]; /*C2H6 */
    W += c[17]*molecular_weights[17]; /*CH2CO */
    W += c[18]*molecular_weights[18]; /*HOCHO */
    W += c[19]*molecular_weights[19]; /*CH3O2 */
    W += c[20]*molecular_weights[20]; /*C2H2 */
    W += c[21]*molecular_weights[21]; /*C3H6 */
    W += c[22]*molecular_weights[22]; /*C2H4 */
    W += c[23]*molecular_weights[23]; /*CH3CO */
    W += c[24]*molecular_weights[24]; /*C3H4XA */
    W += c[25]*molecular_weights[25]; /*C3H5XA */
    W += c[26]*molecular_weights[26]; /*NXC3H7O2 */
    W += c[27]*molecular_weights[27]; /*C4H6 */
    W += c[28]*molecular_weights[28]; /*C4H7 */
    W += c[29]*molecular_weights[29]; /*C4H8X1 */
    W += c[30]*molecular_weights[30]; /*C7H15X2 */
    W += c[31]*molecular_weights[31]; /*NXC7H16 */

    for (id = 0; id < 32; ++id) {
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
    double tmp[32];

    for (int i = 0; i < 32; i++)
    {
        tmp[i] = y[i]*imw[i];
    }
    for (int i = 0; i < 32; i++)
    {
        YOW += tmp[i];
    }

    double YOWINV = 1.0/YOW;

    for (int i = 0; i < 32; i++)
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
    for (int i = 0; i < 32; i++)
    {
        c[i] = y[i]*imw[i];
    }
    for (int i = 0; i < 32; i++)
    {
        YOW += c[i];
    }

    /*PW/RT (see Eq. 7) */
    PWORT = (*P)/(YOW * 8.31446e+07 * (*T)); 
    /*Now compute conversion */

    for (int i = 0; i < 32; i++)
    {
        c[i] = PWORT * y[i] * imw[i];
    }
    return;
}


/*convert y[species] (mass fracs) to c[species] (molar conc) */
AMREX_GPU_HOST_DEVICE void CKYTCR(double *  rho, double *  T, double *  y,  double *  c)
{
    for (int i = 0; i < 32; i++)
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
    XW += x[12]*14.027090; /*CH2GSG */
    XW += x[13]*15.035060; /*CH3 */
    XW += x[14]*16.043030; /*CH4 */
    XW += x[15]*32.042430; /*CH3OH */
    XW += x[16]*30.070120; /*C2H6 */
    XW += x[17]*42.037640; /*CH2CO */
    XW += x[18]*46.025890; /*HOCHO */
    XW += x[19]*47.033860; /*CH3O2 */
    XW += x[20]*26.038240; /*C2H2 */
    XW += x[21]*42.081270; /*C3H6 */
    XW += x[22]*28.054180; /*C2H4 */
    XW += x[23]*43.045610; /*CH3CO */
    XW += x[24]*40.065330; /*C3H4XA */
    XW += x[25]*41.073300; /*C3H5XA */
    XW += x[26]*75.088040; /*NXC3H7O2 */
    XW += x[27]*54.092420; /*C4H6 */
    XW += x[28]*55.100390; /*C4H7 */
    XW += x[29]*56.108360; /*C4H8X1 */
    XW += x[30]*99.197600; /*C7H15X2 */
    XW += x[31]*100.205570; /*NXC7H16 */

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
    y[12] = x[12]*14.027090*XWinv; 
    y[13] = x[13]*15.035060*XWinv; 
    y[14] = x[14]*16.043030*XWinv; 
    y[15] = x[15]*32.042430*XWinv; 
    y[16] = x[16]*30.070120*XWinv; 
    y[17] = x[17]*42.037640*XWinv; 
    y[18] = x[18]*46.025890*XWinv; 
    y[19] = x[19]*47.033860*XWinv; 
    y[20] = x[20]*26.038240*XWinv; 
    y[21] = x[21]*42.081270*XWinv; 
    y[22] = x[22]*28.054180*XWinv; 
    y[23] = x[23]*43.045610*XWinv; 
    y[24] = x[24]*40.065330*XWinv; 
    y[25] = x[25]*41.073300*XWinv; 
    y[26] = x[26]*75.088040*XWinv; 
    y[27] = x[27]*54.092420*XWinv; 
    y[28] = x[28]*55.100390*XWinv; 
    y[29] = x[29]*56.108360*XWinv; 
    y[30] = x[30]*99.197600*XWinv; 
    y[31] = x[31]*100.205570*XWinv; 

    return;
}


/*convert x[species] (mole fracs) to c[species] (molar conc) */
void CKXTCP(double *  P, double *  T, double *  x,  double *  c)
{
    int id; /*loop counter */
    double PORT = (*P)/(8.31446e+07 * (*T)); /*P/RT */

    /*Compute conversion, see Eq 10 */
    for (id = 0; id < 32; ++id) {
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
    XW += x[12]*14.027090; /*CH2GSG */
    XW += x[13]*15.035060; /*CH3 */
    XW += x[14]*16.043030; /*CH4 */
    XW += x[15]*32.042430; /*CH3OH */
    XW += x[16]*30.070120; /*C2H6 */
    XW += x[17]*42.037640; /*CH2CO */
    XW += x[18]*46.025890; /*HOCHO */
    XW += x[19]*47.033860; /*CH3O2 */
    XW += x[20]*26.038240; /*C2H2 */
    XW += x[21]*42.081270; /*C3H6 */
    XW += x[22]*28.054180; /*C2H4 */
    XW += x[23]*43.045610; /*CH3CO */
    XW += x[24]*40.065330; /*C3H4XA */
    XW += x[25]*41.073300; /*C3H5XA */
    XW += x[26]*75.088040; /*NXC3H7O2 */
    XW += x[27]*54.092420; /*C4H6 */
    XW += x[28]*55.100390; /*C4H7 */
    XW += x[29]*56.108360; /*C4H8X1 */
    XW += x[30]*99.197600; /*C7H15X2 */
    XW += x[31]*100.205570; /*NXC7H16 */
    ROW = (*rho) / XW;

    /*Compute conversion, see Eq 11 */
    for (id = 0; id < 32; ++id) {
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
    for (id = 0; id < 32; ++id) {
        sumC += c[id];
    }

    /* See Eq 13  */
    double sumCinv = 1.0/sumC;
    for (id = 0; id < 32; ++id) {
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
    CW += c[12]*14.027090; /*CH2GSG */
    CW += c[13]*15.035060; /*CH3 */
    CW += c[14]*16.043030; /*CH4 */
    CW += c[15]*32.042430; /*CH3OH */
    CW += c[16]*30.070120; /*C2H6 */
    CW += c[17]*42.037640; /*CH2CO */
    CW += c[18]*46.025890; /*HOCHO */
    CW += c[19]*47.033860; /*CH3O2 */
    CW += c[20]*26.038240; /*C2H2 */
    CW += c[21]*42.081270; /*C3H6 */
    CW += c[22]*28.054180; /*C2H4 */
    CW += c[23]*43.045610; /*CH3CO */
    CW += c[24]*40.065330; /*C3H4XA */
    CW += c[25]*41.073300; /*C3H5XA */
    CW += c[26]*75.088040; /*NXC3H7O2 */
    CW += c[27]*54.092420; /*C4H6 */
    CW += c[28]*55.100390; /*C4H7 */
    CW += c[29]*56.108360; /*C4H8X1 */
    CW += c[30]*99.197600; /*C7H15X2 */
    CW += c[31]*100.205570; /*NXC7H16 */
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
    y[12] = c[12]*14.027090*CWinv; 
    y[13] = c[13]*15.035060*CWinv; 
    y[14] = c[14]*16.043030*CWinv; 
    y[15] = c[15]*32.042430*CWinv; 
    y[16] = c[16]*30.070120*CWinv; 
    y[17] = c[17]*42.037640*CWinv; 
    y[18] = c[18]*46.025890*CWinv; 
    y[19] = c[19]*47.033860*CWinv; 
    y[20] = c[20]*26.038240*CWinv; 
    y[21] = c[21]*42.081270*CWinv; 
    y[22] = c[22]*28.054180*CWinv; 
    y[23] = c[23]*43.045610*CWinv; 
    y[24] = c[24]*40.065330*CWinv; 
    y[25] = c[25]*41.073300*CWinv; 
    y[26] = c[26]*75.088040*CWinv; 
    y[27] = c[27]*54.092420*CWinv; 
    y[28] = c[28]*55.100390*CWinv; 
    y[29] = c[29]*56.108360*CWinv; 
    y[30] = c[30]*99.197600*CWinv; 
    y[31] = c[31]*100.205570*CWinv; 

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
    cvms[12] *= 5.927432288630956e+06; /*CH2GSG */
    cvms[13] *= 5.530049509714786e+06; /*CH3 */
    cvms[14] *= 5.182601178301878e+06; /*CH4 */
    cvms[15] *= 2.594828987112788e+06; /*CH3OH */
    cvms[16] *= 2.765024754857393e+06; /*C2H6 */
    cvms[17] *= 1.977861416138784e+06; /*CH2CO */
    cvms[18] *= 1.806475142176118e+06; /*HOCHO */
    cvms[19] *= 1.767761059405552e+06; /*CH3O2 */
    cvms[20] *= 3.193173815954242e+06; /*C2H2 */
    cvms[21] *= 1.975810762876985e+06; /*C3H6 */
    cvms[22] *= 2.963716144315478e+06; /*C2H4 */
    cvms[23] *= 1.931547170118681e+06; /*CH3CO */
    cvms[24] *= 2.075226291198210e+06; /*C3H4XA */
    cvms[25] *= 2.024298660724422e+06; /*C3H5XA */
    cvms[26] *= 1.107295198829699e+06; /*NXC3H7O2 */
    cvms[27] *= 1.537084607816260e+06; /*C4H6 */
    cvms[28] *= 1.508966201174482e+06; /*C4H7 */
    cvms[29] *= 1.481858072157739e+06; /*C4H8X1 */
    cvms[30] *= 8.381717519529947e+05; /*C7H15X2 */
    cvms[31] *= 8.297405641376262e+05; /*NXC7H16 */
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
    cpms[12] *= 5.927432288630956e+06; /*CH2GSG */
    cpms[13] *= 5.530049509714786e+06; /*CH3 */
    cpms[14] *= 5.182601178301878e+06; /*CH4 */
    cpms[15] *= 2.594828987112788e+06; /*CH3OH */
    cpms[16] *= 2.765024754857393e+06; /*C2H6 */
    cpms[17] *= 1.977861416138784e+06; /*CH2CO */
    cpms[18] *= 1.806475142176118e+06; /*HOCHO */
    cpms[19] *= 1.767761059405552e+06; /*CH3O2 */
    cpms[20] *= 3.193173815954242e+06; /*C2H2 */
    cpms[21] *= 1.975810762876985e+06; /*C3H6 */
    cpms[22] *= 2.963716144315478e+06; /*C2H4 */
    cpms[23] *= 1.931547170118681e+06; /*CH3CO */
    cpms[24] *= 2.075226291198210e+06; /*C3H4XA */
    cpms[25] *= 2.024298660724422e+06; /*C3H5XA */
    cpms[26] *= 1.107295198829699e+06; /*NXC3H7O2 */
    cpms[27] *= 1.537084607816260e+06; /*C4H6 */
    cpms[28] *= 1.508966201174482e+06; /*C4H7 */
    cpms[29] *= 1.481858072157739e+06; /*C4H8X1 */
    cpms[30] *= 8.381717519529947e+05; /*C7H15X2 */
    cpms[31] *= 8.297405641376262e+05; /*NXC7H16 */
}


/*Returns internal energy in mass units (Eq 30.) */
AMREX_GPU_HOST_DEVICE void CKUMS(double *  T,  double *  ums)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446e+07*tT; /*R*T */
    speciesInternalEnergy(ums, tc);
    for (int i = 0; i < 32; i++)
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
    for (int i = 0; i < 32; i++)
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
    for (int i = 0; i < 32; i++)
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
    for (int i = 0; i < 32; i++)
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
    sms[12] *= 5.927432288630956e+06; /*CH2GSG */
    sms[13] *= 5.530049509714786e+06; /*CH3 */
    sms[14] *= 5.182601178301878e+06; /*CH4 */
    sms[15] *= 2.594828987112788e+06; /*CH3OH */
    sms[16] *= 2.765024754857393e+06; /*C2H6 */
    sms[17] *= 1.977861416138784e+06; /*CH2CO */
    sms[18] *= 1.806475142176118e+06; /*HOCHO */
    sms[19] *= 1.767761059405552e+06; /*CH3O2 */
    sms[20] *= 3.193173815954242e+06; /*C2H2 */
    sms[21] *= 1.975810762876985e+06; /*C3H6 */
    sms[22] *= 2.963716144315478e+06; /*C2H4 */
    sms[23] *= 1.931547170118681e+06; /*CH3CO */
    sms[24] *= 2.075226291198210e+06; /*C3H4XA */
    sms[25] *= 2.024298660724422e+06; /*C3H5XA */
    sms[26] *= 1.107295198829699e+06; /*NXC3H7O2 */
    sms[27] *= 1.537084607816260e+06; /*C4H6 */
    sms[28] *= 1.508966201174482e+06; /*C4H7 */
    sms[29] *= 1.481858072157739e+06; /*C4H8X1 */
    sms[30] *= 8.381717519529947e+05; /*C7H15X2 */
    sms[31] *= 8.297405641376262e+05; /*NXC7H16 */
}


/*Returns the mean specific heat at CP (Eq. 33) */
void CKCPBL(double *  T, double *  x,  double *  cpbl)
{
    int id; /*loop counter */
    double result = 0; 
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double cpor[32]; /* temporary storage */
    cp_R(cpor, tc);

    /*perform dot product */
    for (id = 0; id < 32; ++id) {
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
    double cpor[32], tresult[32]; /* temporary storage */
    cp_R(cpor, tc);
    for (int i = 0; i < 32; i++)
    {
        tresult[i] = cpor[i]*y[i]*imw[i];

    }
    for (int i = 0; i < 32; i++)
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
    double cvor[32]; /* temporary storage */
    cv_R(cvor, tc);

    /*perform dot product */
    for (id = 0; id < 32; ++id) {
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
    double cvor[32]; /* temporary storage */
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
    result += cvor[12]*y[12]*imw[12]; /*CH2GSG */
    result += cvor[13]*y[13]*imw[13]; /*CH3 */
    result += cvor[14]*y[14]*imw[14]; /*CH4 */
    result += cvor[15]*y[15]*imw[15]; /*CH3OH */
    result += cvor[16]*y[16]*imw[16]; /*C2H6 */
    result += cvor[17]*y[17]*imw[17]; /*CH2CO */
    result += cvor[18]*y[18]*imw[18]; /*HOCHO */
    result += cvor[19]*y[19]*imw[19]; /*CH3O2 */
    result += cvor[20]*y[20]*imw[20]; /*C2H2 */
    result += cvor[21]*y[21]*imw[21]; /*C3H6 */
    result += cvor[22]*y[22]*imw[22]; /*C2H4 */
    result += cvor[23]*y[23]*imw[23]; /*CH3CO */
    result += cvor[24]*y[24]*imw[24]; /*C3H4XA */
    result += cvor[25]*y[25]*imw[25]; /*C3H5XA */
    result += cvor[26]*y[26]*imw[26]; /*NXC3H7O2 */
    result += cvor[27]*y[27]*imw[27]; /*C4H6 */
    result += cvor[28]*y[28]*imw[28]; /*C4H7 */
    result += cvor[29]*y[29]*imw[29]; /*C4H8X1 */
    result += cvor[30]*y[30]*imw[30]; /*C7H15X2 */
    result += cvor[31]*y[31]*imw[31]; /*NXC7H16 */

    *cvbs = result * 8.31446e+07;
}


/*Returns the mean enthalpy of the mixture in molar units */
void CKHBML(double *  T, double *  x,  double *  hbml)
{
    int id; /*loop counter */
    double result = 0; 
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double hml[32]; /* temporary storage */
    double RT = 8.31446e+07*tT; /*R*T */
    speciesEnthalpy(hml, tc);

    /*perform dot product */
    for (id = 0; id < 32; ++id) {
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
    double hml[32], tmp[32]; /* temporary storage */
    double RT = 8.31446e+07*tT; /*R*T */
    speciesEnthalpy(hml, tc);
    int id;
    for (id = 0; id < 32; ++id) {
        tmp[id] = y[id]*hml[id]*imw[id];
    }
    for (id = 0; id < 32; ++id) {
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
    double uml[32]; /* temporary energy array */
    double RT = 8.31446e+07*tT; /*R*T */
    speciesInternalEnergy(uml, tc);

    /*perform dot product */
    for (id = 0; id < 32; ++id) {
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
    double ums[32]; /* temporary energy array */
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
    result += y[12]*ums[12]*imw[12]; /*CH2GSG */
    result += y[13]*ums[13]*imw[13]; /*CH3 */
    result += y[14]*ums[14]*imw[14]; /*CH4 */
    result += y[15]*ums[15]*imw[15]; /*CH3OH */
    result += y[16]*ums[16]*imw[16]; /*C2H6 */
    result += y[17]*ums[17]*imw[17]; /*CH2CO */
    result += y[18]*ums[18]*imw[18]; /*HOCHO */
    result += y[19]*ums[19]*imw[19]; /*CH3O2 */
    result += y[20]*ums[20]*imw[20]; /*C2H2 */
    result += y[21]*ums[21]*imw[21]; /*C3H6 */
    result += y[22]*ums[22]*imw[22]; /*C2H4 */
    result += y[23]*ums[23]*imw[23]; /*CH3CO */
    result += y[24]*ums[24]*imw[24]; /*C3H4XA */
    result += y[25]*ums[25]*imw[25]; /*C3H5XA */
    result += y[26]*ums[26]*imw[26]; /*NXC3H7O2 */
    result += y[27]*ums[27]*imw[27]; /*C4H6 */
    result += y[28]*ums[28]*imw[28]; /*C4H7 */
    result += y[29]*ums[29]*imw[29]; /*C4H8X1 */
    result += y[30]*ums[30]*imw[30]; /*C7H15X2 */
    result += y[31]*ums[31]*imw[31]; /*NXC7H16 */

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
    double sor[32]; /* temporary storage */
    speciesEntropy(sor, tc);

    /*Compute Eq 42 */
    for (id = 0; id < 32; ++id) {
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
    double sor[32]; /* temporary storage */
    double x[32]; /* need a ytx conversion */
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
    YOW += y[12]*imw[12]; /*CH2GSG */
    YOW += y[13]*imw[13]; /*CH3 */
    YOW += y[14]*imw[14]; /*CH4 */
    YOW += y[15]*imw[15]; /*CH3OH */
    YOW += y[16]*imw[16]; /*C2H6 */
    YOW += y[17]*imw[17]; /*CH2CO */
    YOW += y[18]*imw[18]; /*HOCHO */
    YOW += y[19]*imw[19]; /*CH3O2 */
    YOW += y[20]*imw[20]; /*C2H2 */
    YOW += y[21]*imw[21]; /*C3H6 */
    YOW += y[22]*imw[22]; /*C2H4 */
    YOW += y[23]*imw[23]; /*CH3CO */
    YOW += y[24]*imw[24]; /*C3H4XA */
    YOW += y[25]*imw[25]; /*C3H5XA */
    YOW += y[26]*imw[26]; /*NXC3H7O2 */
    YOW += y[27]*imw[27]; /*C4H6 */
    YOW += y[28]*imw[28]; /*C4H7 */
    YOW += y[29]*imw[29]; /*C4H8X1 */
    YOW += y[30]*imw[30]; /*C7H15X2 */
    YOW += y[31]*imw[31]; /*NXC7H16 */
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
    x[12] = y[12]/(14.027090*YOW); 
    x[13] = y[13]/(15.035060*YOW); 
    x[14] = y[14]/(16.043030*YOW); 
    x[15] = y[15]/(32.042430*YOW); 
    x[16] = y[16]/(30.070120*YOW); 
    x[17] = y[17]/(42.037640*YOW); 
    x[18] = y[18]/(46.025890*YOW); 
    x[19] = y[19]/(47.033860*YOW); 
    x[20] = y[20]/(26.038240*YOW); 
    x[21] = y[21]/(42.081270*YOW); 
    x[22] = y[22]/(28.054180*YOW); 
    x[23] = y[23]/(43.045610*YOW); 
    x[24] = y[24]/(40.065330*YOW); 
    x[25] = y[25]/(41.073300*YOW); 
    x[26] = y[26]/(75.088040*YOW); 
    x[27] = y[27]/(54.092420*YOW); 
    x[28] = y[28]/(55.100390*YOW); 
    x[29] = y[29]/(56.108360*YOW); 
    x[30] = y[30]/(99.197600*YOW); 
    x[31] = y[31]/(100.205570*YOW); 
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
    double gort[32]; /* temporary storage */
    /*Compute g/RT */
    gibbs(gort, tc);

    /*Compute Eq 44 */
    for (id = 0; id < 32; ++id) {
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
    double gort[32]; /* temporary storage */
    double x[32]; /* need a ytx conversion */
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
    YOW += y[12]*imw[12]; /*CH2GSG */
    YOW += y[13]*imw[13]; /*CH3 */
    YOW += y[14]*imw[14]; /*CH4 */
    YOW += y[15]*imw[15]; /*CH3OH */
    YOW += y[16]*imw[16]; /*C2H6 */
    YOW += y[17]*imw[17]; /*CH2CO */
    YOW += y[18]*imw[18]; /*HOCHO */
    YOW += y[19]*imw[19]; /*CH3O2 */
    YOW += y[20]*imw[20]; /*C2H2 */
    YOW += y[21]*imw[21]; /*C3H6 */
    YOW += y[22]*imw[22]; /*C2H4 */
    YOW += y[23]*imw[23]; /*CH3CO */
    YOW += y[24]*imw[24]; /*C3H4XA */
    YOW += y[25]*imw[25]; /*C3H5XA */
    YOW += y[26]*imw[26]; /*NXC3H7O2 */
    YOW += y[27]*imw[27]; /*C4H6 */
    YOW += y[28]*imw[28]; /*C4H7 */
    YOW += y[29]*imw[29]; /*C4H8X1 */
    YOW += y[30]*imw[30]; /*C7H15X2 */
    YOW += y[31]*imw[31]; /*NXC7H16 */
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
    x[12] = y[12]/(14.027090*YOW); 
    x[13] = y[13]/(15.035060*YOW); 
    x[14] = y[14]/(16.043030*YOW); 
    x[15] = y[15]/(32.042430*YOW); 
    x[16] = y[16]/(30.070120*YOW); 
    x[17] = y[17]/(42.037640*YOW); 
    x[18] = y[18]/(46.025890*YOW); 
    x[19] = y[19]/(47.033860*YOW); 
    x[20] = y[20]/(26.038240*YOW); 
    x[21] = y[21]/(42.081270*YOW); 
    x[22] = y[22]/(28.054180*YOW); 
    x[23] = y[23]/(43.045610*YOW); 
    x[24] = y[24]/(40.065330*YOW); 
    x[25] = y[25]/(41.073300*YOW); 
    x[26] = y[26]/(75.088040*YOW); 
    x[27] = y[27]/(54.092420*YOW); 
    x[28] = y[28]/(55.100390*YOW); 
    x[29] = y[29]/(56.108360*YOW); 
    x[30] = y[30]/(99.197600*YOW); 
    x[31] = y[31]/(100.205570*YOW); 
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
    double aort[32]; /* temporary storage */
    /*Compute g/RT */
    helmholtz(aort, tc);

    /*Compute Eq 44 */
    for (id = 0; id < 32; ++id) {
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
    double aort[32]; /* temporary storage */
    double x[32]; /* need a ytx conversion */
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
    YOW += y[12]*imw[12]; /*CH2GSG */
    YOW += y[13]*imw[13]; /*CH3 */
    YOW += y[14]*imw[14]; /*CH4 */
    YOW += y[15]*imw[15]; /*CH3OH */
    YOW += y[16]*imw[16]; /*C2H6 */
    YOW += y[17]*imw[17]; /*CH2CO */
    YOW += y[18]*imw[18]; /*HOCHO */
    YOW += y[19]*imw[19]; /*CH3O2 */
    YOW += y[20]*imw[20]; /*C2H2 */
    YOW += y[21]*imw[21]; /*C3H6 */
    YOW += y[22]*imw[22]; /*C2H4 */
    YOW += y[23]*imw[23]; /*CH3CO */
    YOW += y[24]*imw[24]; /*C3H4XA */
    YOW += y[25]*imw[25]; /*C3H5XA */
    YOW += y[26]*imw[26]; /*NXC3H7O2 */
    YOW += y[27]*imw[27]; /*C4H6 */
    YOW += y[28]*imw[28]; /*C4H7 */
    YOW += y[29]*imw[29]; /*C4H8X1 */
    YOW += y[30]*imw[30]; /*C7H15X2 */
    YOW += y[31]*imw[31]; /*NXC7H16 */
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
    x[12] = y[12]/(14.027090*YOW); 
    x[13] = y[13]/(15.035060*YOW); 
    x[14] = y[14]/(16.043030*YOW); 
    x[15] = y[15]/(32.042430*YOW); 
    x[16] = y[16]/(30.070120*YOW); 
    x[17] = y[17]/(42.037640*YOW); 
    x[18] = y[18]/(46.025890*YOW); 
    x[19] = y[19]/(47.033860*YOW); 
    x[20] = y[20]/(26.038240*YOW); 
    x[21] = y[21]/(42.081270*YOW); 
    x[22] = y[22]/(28.054180*YOW); 
    x[23] = y[23]/(43.045610*YOW); 
    x[24] = y[24]/(40.065330*YOW); 
    x[25] = y[25]/(41.073300*YOW); 
    x[26] = y[26]/(75.088040*YOW); 
    x[27] = y[27]/(54.092420*YOW); 
    x[28] = y[28]/(55.100390*YOW); 
    x[29] = y[29]/(56.108360*YOW); 
    x[30] = y[30]/(99.197600*YOW); 
    x[31] = y[31]/(100.205570*YOW); 
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
    /*Scale by RT/W */
    *abms = result * RT * YOW;
}


/*compute the production rate for each species */
AMREX_GPU_HOST_DEVICE void CKWC(double *  T, double *  C,  double *  wdot)
{
    int id; /*loop counter */

    /*convert to SI */
    for (id = 0; id < 32; ++id) {
        C[id] *= 1.0e6;
    }

    /*convert to chemkin units */
    productionRate(wdot, C, *T);

    /*convert to chemkin units */
    for (id = 0; id < 32; ++id) {
        C[id] *= 1.0e-6;
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the molar production rate of species */
/*Given P, T, and mass fractions */
void CKWYP(double *  P, double *  T, double *  y,  double *  wdot)
{
    int id; /*loop counter */
    double c[32]; /*temporary storage */
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
    YOW += y[12]*imw[12]; /*CH2GSG */
    YOW += y[13]*imw[13]; /*CH3 */
    YOW += y[14]*imw[14]; /*CH4 */
    YOW += y[15]*imw[15]; /*CH3OH */
    YOW += y[16]*imw[16]; /*C2H6 */
    YOW += y[17]*imw[17]; /*CH2CO */
    YOW += y[18]*imw[18]; /*HOCHO */
    YOW += y[19]*imw[19]; /*CH3O2 */
    YOW += y[20]*imw[20]; /*C2H2 */
    YOW += y[21]*imw[21]; /*C3H6 */
    YOW += y[22]*imw[22]; /*C2H4 */
    YOW += y[23]*imw[23]; /*CH3CO */
    YOW += y[24]*imw[24]; /*C3H4XA */
    YOW += y[25]*imw[25]; /*C3H5XA */
    YOW += y[26]*imw[26]; /*NXC3H7O2 */
    YOW += y[27]*imw[27]; /*C4H6 */
    YOW += y[28]*imw[28]; /*C4H7 */
    YOW += y[29]*imw[29]; /*C4H8X1 */
    YOW += y[30]*imw[30]; /*C7H15X2 */
    YOW += y[31]*imw[31]; /*NXC7H16 */
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

    /*convert to chemkin units */
    productionRate(wdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 32; ++id) {
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the molar production rate of species */
/*Given P, T, and mole fractions */
void CKWXP(double *  P, double *  T, double *  x,  double *  wdot)
{
    int id; /*loop counter */
    double c[32]; /*temporary storage */
    double PORT = 1e6 * (*P)/(8.31446e+07 * (*T)); /*1e6 * P/RT so c goes to SI units */

    /*Compute conversion, see Eq 10 */
    for (id = 0; id < 32; ++id) {
        c[id] = x[id]*PORT;
    }

    /*convert to chemkin units */
    productionRate(wdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 32; ++id) {
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the molar production rate of species */
/*Given rho, T, and mass fractions */
AMREX_GPU_HOST_DEVICE void CKWYR(double *  rho, double *  T, double *  y,  double *  wdot)
{
    int id; /*loop counter */
    double c[32]; /*temporary storage */
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

    /*call productionRate */
    productionRate(wdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 32; ++id) {
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the molar production rate of species */
/*Given rho, T, and mole fractions */
void CKWXR(double *  rho, double *  T, double *  x,  double *  wdot)
{
    int id; /*loop counter */
    double c[32]; /*temporary storage */
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
    XW += x[12]*14.027090; /*CH2GSG */
    XW += x[13]*15.035060; /*CH3 */
    XW += x[14]*16.043030; /*CH4 */
    XW += x[15]*32.042430; /*CH3OH */
    XW += x[16]*30.070120; /*C2H6 */
    XW += x[17]*42.037640; /*CH2CO */
    XW += x[18]*46.025890; /*HOCHO */
    XW += x[19]*47.033860; /*CH3O2 */
    XW += x[20]*26.038240; /*C2H2 */
    XW += x[21]*42.081270; /*C3H6 */
    XW += x[22]*28.054180; /*C2H4 */
    XW += x[23]*43.045610; /*CH3CO */
    XW += x[24]*40.065330; /*C3H4XA */
    XW += x[25]*41.073300; /*C3H5XA */
    XW += x[26]*75.088040; /*NXC3H7O2 */
    XW += x[27]*54.092420; /*C4H6 */
    XW += x[28]*55.100390; /*C4H7 */
    XW += x[29]*56.108360; /*C4H8X1 */
    XW += x[30]*99.197600; /*C7H15X2 */
    XW += x[31]*100.205570; /*NXC7H16 */
    /*Extra 1e6 factor to take c to SI */
    ROW = 1e6*(*rho) / XW;

    /*Compute conversion, see Eq 11 */
    for (id = 0; id < 32; ++id) {
        c[id] = x[id]*ROW;
    }

    /*convert to chemkin units */
    productionRate(wdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 32; ++id) {
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
    for (id = 0; id < kd * 32; ++ id) {
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

    /*CH2GSG */
    ncf[ 12 * kd + 3 ] = 1; /*C */
    ncf[ 12 * kd + 2 ] = 2; /*H */

    /*CH3 */
    ncf[ 13 * kd + 3 ] = 1; /*C */
    ncf[ 13 * kd + 2 ] = 3; /*H */

    /*CH4 */
    ncf[ 14 * kd + 3 ] = 1; /*C */
    ncf[ 14 * kd + 2 ] = 4; /*H */

    /*CH3OH */
    ncf[ 15 * kd + 3 ] = 1; /*C */
    ncf[ 15 * kd + 2 ] = 4; /*H */
    ncf[ 15 * kd + 1 ] = 1; /*O */

    /*C2H6 */
    ncf[ 16 * kd + 3 ] = 2; /*C */
    ncf[ 16 * kd + 2 ] = 6; /*H */

    /*CH2CO */
    ncf[ 17 * kd + 3 ] = 2; /*C */
    ncf[ 17 * kd + 2 ] = 2; /*H */
    ncf[ 17 * kd + 1 ] = 1; /*O */

    /*HOCHO */
    ncf[ 18 * kd + 3 ] = 1; /*C */
    ncf[ 18 * kd + 2 ] = 2; /*H */
    ncf[ 18 * kd + 1 ] = 2; /*O */

    /*CH3O2 */
    ncf[ 19 * kd + 3 ] = 1; /*C */
    ncf[ 19 * kd + 2 ] = 3; /*H */
    ncf[ 19 * kd + 1 ] = 2; /*O */

    /*C2H2 */
    ncf[ 20 * kd + 3 ] = 2; /*C */
    ncf[ 20 * kd + 2 ] = 2; /*H */

    /*C3H6 */
    ncf[ 21 * kd + 3 ] = 3; /*C */
    ncf[ 21 * kd + 2 ] = 6; /*H */

    /*C2H4 */
    ncf[ 22 * kd + 3 ] = 2; /*C */
    ncf[ 22 * kd + 2 ] = 4; /*H */

    /*CH3CO */
    ncf[ 23 * kd + 3 ] = 2; /*C */
    ncf[ 23 * kd + 2 ] = 3; /*H */
    ncf[ 23 * kd + 1 ] = 1; /*O */

    /*C3H4XA */
    ncf[ 24 * kd + 2 ] = 4; /*H */
    ncf[ 24 * kd + 3 ] = 3; /*C */

    /*C3H5XA */
    ncf[ 25 * kd + 3 ] = 3; /*C */
    ncf[ 25 * kd + 2 ] = 5; /*H */

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

    /*C7H15X2 */
    ncf[ 30 * kd + 3 ] = 7; /*C */
    ncf[ 30 * kd + 2 ] = 15; /*H */

    /*NXC7H16 */
    ncf[ 31 * kd + 3 ] = 7; /*C */
    ncf[ 31 * kd + 2 ] = 16; /*H */


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

static double k_f_save_qss[133];
#ifdef _OPENMP
#pragma omp threadprivate(k_f_save)
#endif

static double Kc_save_qss[133];
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

    double qdot, q_f[218], q_r[218];
    double sc_qss[20];
    /* Fill sc_qss here*/
    comp_qss_sc(sc, sc_qss, tc, invT);
    comp_qfqr(q_f, q_r, sc, sc_qss, tc, invT);

    for (int i = 0; i < 32; ++i) {
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
    wdot[13] -= qdot;
    wdot[15] += qdot;

    qdot = q_f[3]-q_r[3];
    wdot[3] -= qdot;
    wdot[13] -= qdot;
    wdot[14] += qdot;

    qdot = q_f[4]-q_r[4];
    wdot[13] -= 2.000000 * qdot;
    wdot[16] += qdot;

    qdot = q_f[5]-q_r[5];
    wdot[10] -= qdot;
    wdot[17] += qdot;

    qdot = q_f[6]-q_r[6];
    wdot[1] -= qdot;
    wdot[9] += qdot;
    wdot[10] -= qdot;

    qdot = q_f[7]-q_r[7];
    wdot[3] += qdot;
    wdot[11] += qdot;

    qdot = q_f[8]-q_r[8];
    wdot[3] += qdot;
    wdot[20] += qdot;

    qdot = q_f[9]-q_r[9];
    wdot[3] -= qdot;
    wdot[22] -= qdot;

    qdot = q_f[10]-q_r[10];
    wdot[10] += qdot;
    wdot[13] += qdot;
    wdot[23] -= qdot;

    qdot = q_f[11]-q_r[11];
    wdot[3] -= qdot;
    wdot[4] -= qdot;
    wdot[5] += qdot;

    qdot = q_f[12]-q_r[12];
    wdot[12] -= qdot;

    qdot = q_f[13]-q_r[13];
    wdot[12] += qdot;

    qdot = q_f[14]-q_r[14];
    wdot[3] += qdot;
    wdot[10] += qdot;

    qdot = q_f[15]-q_r[15];
    wdot[6] += qdot;
    wdot[13] += qdot;
    wdot[19] -= qdot;

    qdot = q_f[16]-q_r[16];
    wdot[6] -= qdot;
    wdot[13] -= qdot;
    wdot[19] += qdot;

    qdot = q_f[17]-q_r[17];
    wdot[11] += qdot;
    wdot[13] += qdot;

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
    wdot[12] -= qdot;

    qdot = q_f[44]-q_r[44];
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[12] -= qdot;
    wdot[13] += qdot;

    qdot = q_f[45]-q_r[45];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[12] += qdot;
    wdot[13] -= qdot;

    qdot = q_f[46]-q_r[46];
    wdot[3] += qdot;
    wdot[4] += qdot;
    wdot[6] -= qdot;
    wdot[10] += qdot;
    wdot[12] -= qdot;

    qdot = q_f[47]-q_r[47];
    wdot[3] += qdot;
    wdot[4] -= qdot;
    wdot[11] += qdot;
    wdot[12] -= qdot;

    qdot = q_f[48]-q_r[48];
    wdot[2] += qdot;
    wdot[4] -= qdot;
    wdot[11] += qdot;
    wdot[13] -= qdot;

    qdot = q_f[49]-q_r[49];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[12] += qdot;
    wdot[13] -= qdot;

    qdot = q_f[50]-q_r[50];
    wdot[4] += qdot;
    wdot[5] -= qdot;
    wdot[12] -= qdot;
    wdot[13] += qdot;

    qdot = q_f[51]-q_r[51];
    wdot[1] -= qdot;
    wdot[3] += qdot;
    wdot[11] += qdot;
    wdot[13] -= qdot;

    qdot = q_f[52]-q_r[52];
    wdot[4] += qdot;
    wdot[7] -= qdot;
    wdot[13] -= qdot;

    qdot = q_f[53]-q_r[53];
    wdot[6] += qdot;
    wdot[7] -= qdot;
    wdot[13] -= qdot;
    wdot[14] += qdot;

    qdot = q_f[54]-q_r[54];
    wdot[4] += qdot;
    wdot[6] -= qdot;
    wdot[11] += qdot;
    wdot[13] -= qdot;

    qdot = q_f[55]-q_r[55];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[13] -= qdot;

    qdot = q_f[56]-q_r[56];
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[13] += qdot;

    qdot = q_f[57]-q_r[57];
    wdot[3] += qdot;
    wdot[13] -= 2.000000 * qdot;

    qdot = q_f[58]-q_r[58];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[13] -= qdot;

    qdot = q_f[59]-q_r[59];
    wdot[4] += qdot;
    wdot[5] -= qdot;
    wdot[13] += qdot;

    qdot = q_f[60]-q_r[60];
    wdot[1] -= qdot;
    wdot[4] += qdot;
    wdot[13] += qdot;
    wdot[14] -= qdot;

    qdot = q_f[61]-q_r[61];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[13] += qdot;
    wdot[14] -= qdot;

    qdot = q_f[62]-q_r[62];
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[13] -= qdot;
    wdot[14] += qdot;

    qdot = q_f[63]-q_r[63];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[13] += qdot;
    wdot[14] -= qdot;

    qdot = q_f[64]-q_r[64];
    wdot[4] += qdot;
    wdot[5] -= qdot;
    wdot[13] -= qdot;
    wdot[14] += qdot;

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
    wdot[13] -= qdot;
    wdot[14] += qdot;

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
    wdot[13] -= qdot;
    wdot[14] += qdot;

    qdot = q_f[77]-q_r[77];
    wdot[11] += qdot;
    wdot[15] += qdot;

    qdot = q_f[78]-q_r[78];
    wdot[6] -= qdot;
    wdot[7] += qdot;
    wdot[11] += qdot;

    qdot = q_f[79]-q_r[79];
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[15] += qdot;

    qdot = q_f[80]-q_r[80];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[15] -= qdot;

    qdot = q_f[81]-q_r[81];
    wdot[9] -= qdot;
    wdot[10] += qdot;
    wdot[11] += qdot;
    wdot[12] -= qdot;

    qdot = q_f[82]-q_r[82];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[4] += qdot;
    wdot[10] += qdot;
    wdot[18] -= qdot;

    qdot = q_f[83]-q_r[83];
    wdot[4] -= qdot;
    wdot[4] += qdot;
    wdot[5] += qdot;
    wdot[10] += qdot;
    wdot[18] -= qdot;

    qdot = q_f[84]-q_r[84];
    wdot[4] += qdot;
    wdot[18] -= qdot;

    qdot = q_f[85]-q_r[85];
    wdot[4] -= qdot;
    wdot[18] += qdot;

    qdot = q_f[86]-q_r[86];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[3] += qdot;
    wdot[9] += qdot;
    wdot[18] -= qdot;

    qdot = q_f[87]-q_r[87];
    wdot[3] += qdot;
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[9] += qdot;
    wdot[18] -= qdot;

    qdot = q_f[88]-q_r[88];
    wdot[6] += qdot;
    wdot[19] -= 2.000000 * qdot;

    qdot = q_f[89]-q_r[89];
    wdot[13] -= qdot;
    wdot[19] -= qdot;

    qdot = q_f[90]-q_r[90];
    wdot[6] += qdot;
    wdot[11] += qdot;
    wdot[15] += qdot;
    wdot[19] -= 2.000000 * qdot;

    qdot = q_f[91]-q_r[91];
    wdot[6] += qdot;
    wdot[7] -= qdot;
    wdot[19] -= qdot;

    qdot = q_f[92]-q_r[92];
    wdot[4] += qdot;

    qdot = q_f[93]-q_r[93];
    wdot[1] -= qdot;
    wdot[10] += qdot;
    wdot[20] -= qdot;

    qdot = q_f[94]-q_r[94];
    wdot[1] -= qdot;
    wdot[3] += qdot;
    wdot[20] -= qdot;

    qdot = q_f[95]-q_r[95];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[20] += qdot;

    qdot = q_f[96]-q_r[96];
    wdot[1] += qdot;
    wdot[6] -= qdot;

    qdot = q_f[97]-q_r[97];
    wdot[13] -= qdot;
    wdot[21] += qdot;

    qdot = q_f[98]-q_r[98];
    wdot[6] -= qdot;
    wdot[7] += qdot;
    wdot[20] += qdot;

    qdot = q_f[99]-q_r[99];
    wdot[6] -= qdot;
    wdot[11] += qdot;

    qdot = q_f[100]-q_r[100];
    wdot[13] -= qdot;
    wdot[14] += qdot;
    wdot[22] -= qdot;

    qdot = q_f[101]-q_r[101];
    wdot[1] -= qdot;
    wdot[13] += qdot;
    wdot[22] -= qdot;

    qdot = q_f[102]-q_r[102];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[22] -= qdot;

    qdot = q_f[103]-q_r[103];
    wdot[1] -= qdot;
    wdot[3] += qdot;
    wdot[22] -= qdot;

    qdot = q_f[104]-q_r[104];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[22] -= qdot;

    qdot = q_f[105]-q_r[105];
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[22] += qdot;

    qdot = q_f[106]-q_r[106];
    wdot[3] -= qdot;
    wdot[16] += qdot;

    qdot = q_f[107]-q_r[107];
    wdot[19] -= qdot;

    qdot = q_f[108]-q_r[108];
    wdot[4] += qdot;
    wdot[7] -= qdot;

    qdot = q_f[109]-q_r[109];
    wdot[6] -= qdot;
    wdot[7] += qdot;
    wdot[22] += qdot;

    qdot = q_f[110]-q_r[110];
    wdot[1] -= qdot;
    wdot[4] += qdot;
    wdot[16] -= qdot;

    qdot = q_f[111]-q_r[111];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[16] -= qdot;

    qdot = q_f[112]-q_r[112];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[16] -= qdot;

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
    wdot[12] += qdot;

    qdot = q_f[117]-q_r[117];
    wdot[3] += qdot;
    wdot[10] -= qdot;
    wdot[12] -= qdot;

    qdot = q_f[118]-q_r[118];
    wdot[1] -= qdot;
    wdot[4] += qdot;
    wdot[17] -= qdot;

    qdot = q_f[119]-q_r[119];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[17] -= qdot;

    qdot = q_f[120]-q_r[120];
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[17] += qdot;

    qdot = q_f[121]-q_r[121];
    wdot[3] -= qdot;
    wdot[10] += qdot;
    wdot[13] += qdot;
    wdot[17] -= qdot;

    qdot = q_f[122]-q_r[122];
    wdot[1] -= qdot;
    wdot[9] += qdot;
    wdot[17] -= qdot;

    qdot = q_f[123]-q_r[123];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[17] -= qdot;

    qdot = q_f[124]-q_r[124];
    wdot[4] += qdot;
    wdot[6] -= qdot;
    wdot[10] += qdot;
    wdot[11] += qdot;

    qdot = q_f[125]-q_r[125];
    wdot[3] += qdot;
    wdot[17] += qdot;

    qdot = q_f[126]-q_r[126];
    wdot[3] -= qdot;
    wdot[17] -= qdot;

    qdot = q_f[127]-q_r[127];
    wdot[6] += qdot;

    qdot = q_f[128]-q_r[128];
    wdot[6] -= qdot;

    qdot = q_f[129]-q_r[129];
    wdot[7] += qdot;
    wdot[22] += qdot;

    qdot = q_f[130]-q_r[130];
    wdot[3] += qdot;
    wdot[6] -= qdot;
    wdot[10] += qdot;

    qdot = q_f[131]-q_r[131];
    wdot[4] -= qdot;
    wdot[20] += qdot;

    qdot = q_f[132]-q_r[132];
    wdot[6] -= qdot;
    wdot[17] += qdot;

    qdot = q_f[133]-q_r[133];
    wdot[6] += qdot;
    wdot[7] -= qdot;
    wdot[24] += qdot;

    qdot = q_f[134]-q_r[134];
    wdot[2] += qdot;
    wdot[3] -= qdot;

    qdot = q_f[135]-q_r[135];
    wdot[4] -= qdot;
    wdot[5] += qdot;

    qdot = q_f[136]-q_r[136];
    wdot[4] += qdot;
    wdot[5] -= qdot;

    qdot = q_f[137]-q_r[137];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[24] -= qdot;

    qdot = q_f[138]-q_r[138];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[24] -= qdot;

    qdot = q_f[139]-q_r[139];
    wdot[1] -= qdot;
    wdot[10] += qdot;
    wdot[22] += qdot;
    wdot[24] -= qdot;

    qdot = q_f[140]-q_r[140];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[24] += qdot;
    wdot[25] -= qdot;

    qdot = q_f[141]-q_r[141];
    wdot[6] += qdot;
    wdot[7] -= qdot;
    wdot[21] += qdot;
    wdot[25] -= qdot;

    qdot = q_f[142]-q_r[142];
    wdot[3] -= qdot;
    wdot[21] += qdot;
    wdot[25] -= qdot;

    qdot = q_f[143]-q_r[143];
    wdot[13] += qdot;
    wdot[20] += qdot;
    wdot[25] -= qdot;

    qdot = q_f[144]-q_r[144];
    wdot[3] += qdot;
    wdot[24] += qdot;
    wdot[25] -= qdot;

    qdot = q_f[145]-q_r[145];
    wdot[3] -= qdot;
    wdot[24] -= qdot;
    wdot[25] += qdot;

    qdot = q_f[146]-q_r[146];
    wdot[11] -= qdot;
    wdot[21] += qdot;
    wdot[25] -= qdot;

    qdot = q_f[147]-q_r[147];
    wdot[21] += qdot;
    wdot[24] += qdot;
    wdot[25] -= 2.000000 * qdot;

    qdot = q_f[148]-q_r[148];
    wdot[3] -= qdot;
    wdot[13] += qdot;
    wdot[21] -= qdot;
    wdot[22] += qdot;

    qdot = q_f[149]-q_r[149];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[21] -= qdot;
    wdot[25] += qdot;

    qdot = q_f[150]-q_r[150];
    wdot[1] -= qdot;
    wdot[21] -= qdot;

    qdot = q_f[151]-q_r[151];
    wdot[1] -= qdot;
    wdot[4] += qdot;
    wdot[21] -= qdot;
    wdot[25] += qdot;

    qdot = q_f[152]-q_r[152];
    wdot[1] -= qdot;
    wdot[3] += qdot;
    wdot[13] += qdot;
    wdot[17] += qdot;
    wdot[21] -= qdot;

    qdot = q_f[153]-q_r[153];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[21] -= qdot;
    wdot[25] += qdot;

    qdot = q_f[154]-q_r[154];
    wdot[6] -= qdot;
    wdot[7] += qdot;
    wdot[21] += qdot;

    qdot = q_f[155]-q_r[155];
    wdot[13] += qdot;
    wdot[22] += qdot;

    qdot = q_f[156]-q_r[156];
    wdot[13] -= qdot;
    wdot[22] -= qdot;

    qdot = q_f[157]-q_r[157];
    wdot[3] += qdot;
    wdot[21] += qdot;

    qdot = q_f[158]-q_r[158];
    wdot[3] -= qdot;
    wdot[21] -= qdot;

    qdot = q_f[159]-q_r[159];
    wdot[6] += qdot;
    wdot[26] -= qdot;

    qdot = q_f[160]-q_r[160];
    wdot[6] -= qdot;
    wdot[26] += qdot;

    qdot = q_f[161]-q_r[161];
    wdot[27] -= qdot;

    qdot = q_f[162]-q_r[162];
    wdot[27] += qdot;

    qdot = q_f[163]-q_r[163];
    wdot[4] -= qdot;
    wdot[11] += qdot;
    wdot[25] += qdot;
    wdot[27] -= qdot;

    qdot = q_f[164]-q_r[164];
    wdot[4] -= qdot;
    wdot[17] += qdot;
    wdot[27] -= qdot;

    qdot = q_f[165]-q_r[165];
    wdot[1] -= qdot;
    wdot[17] += qdot;
    wdot[22] += qdot;
    wdot[27] -= qdot;

    qdot = q_f[166]-q_r[166];
    wdot[3] -= qdot;
    wdot[22] += qdot;
    wdot[27] -= qdot;

    qdot = q_f[167]-q_r[167];
    wdot[1] -= qdot;
    wdot[11] += qdot;
    wdot[24] += qdot;
    wdot[27] -= qdot;

    qdot = q_f[168]-q_r[168];
    wdot[3] -= qdot;
    wdot[28] -= qdot;
    wdot[29] += qdot;

    qdot = q_f[169]-q_r[169];
    wdot[21] += qdot;
    wdot[25] -= qdot;
    wdot[27] += qdot;
    wdot[28] -= qdot;

    qdot = q_f[170]-q_r[170];
    wdot[16] += qdot;
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
    wdot[13] -= qdot;
    wdot[14] += qdot;
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
    wdot[22] += qdot;
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
    wdot[29] -= qdot;

    qdot = q_f[180]-q_r[180];
    wdot[4] -= qdot;
    wdot[16] += qdot;
    wdot[23] += qdot;
    wdot[29] -= qdot;

    qdot = q_f[181]-q_r[181];
    wdot[1] -= qdot;
    wdot[23] += qdot;
    wdot[29] -= qdot;

    qdot = q_f[182]-q_r[182];
    wdot[1] -= qdot;
    wdot[11] += qdot;
    wdot[21] += qdot;
    wdot[29] -= qdot;

    qdot = q_f[183]-q_r[183];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[28] += qdot;
    wdot[29] -= qdot;

    qdot = q_f[184]-q_r[184];
    wdot[13] += qdot;
    wdot[25] += qdot;
    wdot[29] -= qdot;

    qdot = q_f[185]-q_r[185];
    wdot[13] -= qdot;
    wdot[25] -= qdot;
    wdot[29] += qdot;

    qdot = q_f[186]-q_r[186];
    wdot[3] += qdot;
    wdot[29] += qdot;

    qdot = q_f[187]-q_r[187];
    wdot[3] -= qdot;
    wdot[29] -= qdot;

    qdot = q_f[188]-q_r[188];
    wdot[22] += qdot;

    qdot = q_f[189]-q_r[189];
    wdot[6] += qdot;

    qdot = q_f[190]-q_r[190];
    wdot[6] -= qdot;

    qdot = q_f[191]-q_r[191];
    wdot[13] += qdot;
    wdot[27] += qdot;

    qdot = q_f[192]-q_r[192];
    wdot[22] += qdot;
    wdot[25] += qdot;

    qdot = q_f[193]-q_r[193];
    wdot[4] -= qdot;
    wdot[5] += qdot;

    qdot = q_f[194]-q_r[194];
    wdot[2] += qdot;
    wdot[3] -= qdot;

    qdot = q_f[195]-q_r[195];
    wdot[25] += qdot;

    qdot = q_f[196]-q_r[196];
    wdot[25] -= qdot;

    qdot = q_f[197]-q_r[197];
    wdot[1] -= qdot;
    wdot[4] += qdot;

    qdot = q_f[198]-q_r[198];
    wdot[21] += qdot;

    qdot = q_f[199]-q_r[199];
    wdot[22] += qdot;

    qdot = q_f[200]-q_r[200];
    wdot[3] += qdot;

    qdot = q_f[201]-q_r[201];
    wdot[25] += qdot;

    qdot = q_f[202]-q_r[202];
    wdot[4] -= qdot;
    wdot[11] += qdot;

    qdot = q_f[203]-q_r[203];
    wdot[13] += qdot;
    wdot[30] -= qdot;

    qdot = q_f[204]-q_r[204];
    wdot[21] += qdot;
    wdot[30] -= qdot;

    qdot = q_f[205]-q_r[205];
    wdot[29] += qdot;
    wdot[30] -= qdot;

    qdot = q_f[206]-q_r[206];
    wdot[22] += qdot;
    wdot[30] -= qdot;

    qdot = q_f[207]-q_r[207];
    wdot[30] -= qdot;

    qdot = q_f[208]-q_r[208];
    wdot[6] += qdot;
    wdot[7] -= qdot;
    wdot[30] -= qdot;
    wdot[31] += qdot;

    qdot = q_f[209]-q_r[209];
    wdot[19] -= qdot;
    wdot[30] += qdot;
    wdot[31] -= qdot;

    qdot = q_f[210]-q_r[210];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[30] += qdot;
    wdot[31] -= qdot;

    qdot = q_f[211]-q_r[211];
    wdot[31] -= qdot;

    qdot = q_f[212]-q_r[212];
    wdot[7] -= qdot;
    wdot[8] += qdot;
    wdot[30] += qdot;
    wdot[31] -= qdot;

    qdot = q_f[213]-q_r[213];
    wdot[31] -= qdot;

    qdot = q_f[214]-q_r[214];
    wdot[15] += qdot;
    wdot[30] += qdot;
    wdot[31] -= qdot;

    qdot = q_f[215]-q_r[215];
    wdot[1] -= qdot;
    wdot[4] += qdot;
    wdot[30] += qdot;
    wdot[31] -= qdot;

    qdot = q_f[216]-q_r[216];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[30] += qdot;
    wdot[31] -= qdot;

    qdot = q_f[217]-q_r[217];
    wdot[13] -= qdot;
    wdot[14] += qdot;
    wdot[30] += qdot;
    wdot[31] -= qdot;

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
    double g_RT[32], g_RT_qss[20];
    gibbs(g_RT, tc);
    gibbs_qss(g_RT_qss, tc);

    Kc[0] = g_RT[3] + g_RT[6] - g_RT[7];
    Kc[1] = -2.000000*g_RT[4] + g_RT[8];
    Kc[2] = g_RT[4] + g_RT[13] - g_RT[15];
    Kc[3] = g_RT[3] + g_RT[13] - g_RT[14];
    Kc[4] = 2.000000*g_RT[13] - g_RT[16];
    Kc[5] = g_RT[10] - g_RT[17] + g_RT_qss[2];
    Kc[6] = g_RT[1] - g_RT[9] + g_RT[10];
    Kc[7] = -g_RT[3] - g_RT[11] + g_RT_qss[3];
    Kc[8] = -g_RT[3] - g_RT[20] + g_RT_qss[7];
    Kc[9] = g_RT[3] + g_RT[22] - g_RT_qss[4];
    Kc[10] = -g_RT[10] - g_RT[13] + g_RT[23];
    Kc[11] = g_RT[3] + g_RT[4] - g_RT[5];
    Kc[12] = g_RT[12] - g_RT_qss[2];
    Kc[13] = -g_RT[12] + g_RT_qss[2];
    Kc[14] = -g_RT[3] - g_RT[10] + g_RT_qss[1];
    Kc[15] = -g_RT[6] - g_RT[13] + g_RT[19];
    Kc[16] = g_RT[6] + g_RT[13] - g_RT[19];
    Kc[17] = -g_RT[11] - g_RT[13] + g_RT_qss[9];
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
    Kc[43] = -g_RT[2] + g_RT[3] + g_RT[12] - g_RT_qss[0];
    Kc[44] = g_RT[2] - g_RT[3] + g_RT[12] - g_RT[13];
    Kc[45] = -g_RT[2] + g_RT[3] - g_RT[12] + g_RT[13];
    Kc[46] = -g_RT[3] - g_RT[4] + g_RT[6] - g_RT[10] + g_RT[12];
    Kc[47] = -g_RT[3] + g_RT[4] - g_RT[11] + g_RT[12];
    Kc[48] = -g_RT[2] + g_RT[4] - g_RT[11] + g_RT[13];
    Kc[49] = g_RT[4] - g_RT[5] - g_RT[12] + g_RT[13];
    Kc[50] = -g_RT[4] + g_RT[5] + g_RT[12] - g_RT[13];
    Kc[51] = g_RT[1] - g_RT[3] - g_RT[11] + g_RT[13];
    Kc[52] = -g_RT[4] + g_RT[7] + g_RT[13] - g_RT_qss[3];
    Kc[53] = -g_RT[6] + g_RT[7] + g_RT[13] - g_RT[14];
    Kc[54] = -g_RT[4] + g_RT[6] - g_RT[11] + g_RT[13];
    Kc[55] = -g_RT[2] + g_RT[3] + g_RT[13] - g_RT_qss[2];
    Kc[56] = g_RT[2] - g_RT[3] - g_RT[13] + g_RT_qss[2];
    Kc[57] = -g_RT[3] + 2.000000*g_RT[13] - g_RT_qss[4];
    Kc[58] = g_RT[4] - g_RT[5] + g_RT[13] - g_RT_qss[2];
    Kc[59] = -g_RT[4] + g_RT[5] - g_RT[13] + g_RT_qss[2];
    Kc[60] = g_RT[1] - g_RT[4] - g_RT[13] + g_RT[14];
    Kc[61] = -g_RT[2] + g_RT[3] - g_RT[13] + g_RT[14];
    Kc[62] = g_RT[2] - g_RT[3] + g_RT[13] - g_RT[14];
    Kc[63] = g_RT[4] - g_RT[5] - g_RT[13] + g_RT[14];
    Kc[64] = -g_RT[4] + g_RT[5] + g_RT[13] - g_RT[14];
    Kc[65] = -g_RT[3] + g_RT[4] - g_RT[9] + g_RT[10];
    Kc[66] = g_RT[3] - g_RT[4] + g_RT[9] - g_RT[10];
    Kc[67] = g_RT[6] - g_RT[7] - g_RT[10] + g_RT_qss[1];
    Kc[68] = g_RT[1] - g_RT[3] - g_RT[9] + g_RT_qss[1];
    Kc[69] = g_RT[4] - g_RT[5] - g_RT[10] + g_RT_qss[1];
    Kc[70] = -g_RT[2] + g_RT[3] - g_RT[10] + g_RT_qss[1];
    Kc[71] = g_RT[1] - g_RT[4] - g_RT[10] + g_RT_qss[1];
    Kc[72] = -g_RT[10] + g_RT[13] - g_RT[14] + g_RT_qss[1];
    Kc[73] = g_RT[4] - g_RT[5] + g_RT[11] - g_RT_qss[1];
    Kc[74] = g_RT[1] - g_RT[4] + g_RT[11] - g_RT_qss[1];
    Kc[75] = -g_RT[2] + g_RT[3] + g_RT[11] - g_RT_qss[1];
    Kc[76] = g_RT[11] + g_RT[13] - g_RT[14] - g_RT_qss[1];
    Kc[77] = -g_RT[11] - g_RT[15] + 2.000000*g_RT_qss[3];
    Kc[78] = g_RT[6] - g_RT[7] - g_RT[11] + g_RT_qss[3];
    Kc[79] = g_RT[2] - g_RT[3] - g_RT[15] + g_RT_qss[3];
    Kc[80] = g_RT[4] - g_RT[5] + g_RT[15] - g_RT_qss[3];
    Kc[81] = g_RT[9] - g_RT[10] - g_RT[11] + g_RT[12];
    Kc[82] = -g_RT[2] + g_RT[3] - g_RT[4] - g_RT[10] + g_RT[18];
    Kc[83] = g_RT[4] - g_RT[4] - g_RT[5] - g_RT[10] + g_RT[18];
    Kc[84] = -g_RT[4] + g_RT[18] - g_RT_qss[1];
    Kc[85] = g_RT[4] - g_RT[18] + g_RT_qss[1];
    Kc[86] = -g_RT[2] + g_RT[3] - g_RT[3] - g_RT[9] + g_RT[18];
    Kc[87] = -g_RT[3] + g_RT[4] - g_RT[5] - g_RT[9] + g_RT[18];
    Kc[88] = -g_RT[6] + 2.000000*g_RT[19] - 2.000000*g_RT_qss[3];
    Kc[89] = g_RT[13] + g_RT[19] - 2.000000*g_RT_qss[3];
    Kc[90] = -g_RT[6] - g_RT[11] - g_RT[15] + 2.000000*g_RT[19];
    Kc[91] = -g_RT[6] + g_RT[7] + g_RT[19] - g_RT_qss[5];
    Kc[92] = -g_RT[4] - g_RT_qss[3] + g_RT_qss[5];
    Kc[93] = g_RT[1] - g_RT[10] + g_RT[20] - g_RT_qss[2];
    Kc[94] = g_RT[1] - g_RT[3] + g_RT[20] - g_RT_qss[6];
    Kc[95] = -g_RT[2] + g_RT[3] - g_RT[20] + g_RT_qss[7];
    Kc[96] = -g_RT[1] + g_RT[6] + g_RT_qss[7] - g_RT_qss[8];
    Kc[97] = g_RT[13] - g_RT[21] + g_RT_qss[7];
    Kc[98] = g_RT[6] - g_RT[7] - g_RT[20] + g_RT_qss[7];
    Kc[99] = g_RT[6] - g_RT[11] - g_RT_qss[1] + g_RT_qss[7];
    Kc[100] = g_RT[13] - g_RT[14] + g_RT[22] - g_RT_qss[7];
    Kc[101] = g_RT[1] - g_RT[13] + g_RT[22] - g_RT_qss[1];
    Kc[102] = g_RT[4] - g_RT[5] + g_RT[22] - g_RT_qss[7];
    Kc[103] = g_RT[1] - g_RT[3] + g_RT[22] - g_RT_qss[8];
    Kc[104] = -g_RT[2] + g_RT[3] + g_RT[22] - g_RT_qss[7];
    Kc[105] = g_RT[2] - g_RT[3] - g_RT[22] + g_RT_qss[7];
    Kc[106] = g_RT[3] - g_RT[16] + g_RT_qss[4];
    Kc[107] = g_RT[19] - g_RT_qss[3] + g_RT_qss[4] - g_RT_qss[9];
    Kc[108] = -g_RT[4] + g_RT[7] + g_RT_qss[4] - g_RT_qss[9];
    Kc[109] = g_RT[6] - g_RT[7] - g_RT[22] + g_RT_qss[4];
    Kc[110] = g_RT[1] - g_RT[4] + g_RT[16] - g_RT_qss[4];
    Kc[111] = g_RT[4] - g_RT[5] + g_RT[16] - g_RT_qss[4];
    Kc[112] = -g_RT[2] + g_RT[3] + g_RT[16] - g_RT_qss[4];
    Kc[113] = g_RT[1] - g_RT[3] - 2.000000*g_RT[10] + g_RT_qss[6];
    Kc[114] = g_RT[4] - 2.000000*g_RT_qss[1] + g_RT_qss[6];
    Kc[115] = g_RT[6] - g_RT[9] - g_RT_qss[1] + g_RT_qss[6];
    Kc[116] = g_RT[3] - g_RT[10] - g_RT[12] + g_RT_qss[6];
    Kc[117] = -g_RT[3] + g_RT[10] + g_RT[12] - g_RT_qss[6];
    Kc[118] = g_RT[1] - g_RT[4] + g_RT[17] - g_RT_qss[6];
    Kc[119] = -g_RT[2] + g_RT[3] + g_RT[17] - g_RT_qss[6];
    Kc[120] = g_RT[2] - g_RT[3] - g_RT[17] + g_RT_qss[6];
    Kc[121] = g_RT[3] - g_RT[10] - g_RT[13] + g_RT[17];
    Kc[122] = g_RT[1] - g_RT[9] + g_RT[17] - g_RT_qss[2];
    Kc[123] = g_RT[4] - g_RT[5] + g_RT[17] - g_RT_qss[6];
    Kc[124] = -g_RT[4] + g_RT[6] - g_RT[10] - g_RT[11] + g_RT_qss[8];
    Kc[125] = -g_RT[3] - g_RT[17] + g_RT_qss[8];
    Kc[126] = g_RT[3] + g_RT[17] - g_RT_qss[8];
    Kc[127] = -g_RT[6] - g_RT_qss[4] + g_RT_qss[10];
    Kc[128] = g_RT[6] + g_RT_qss[4] - g_RT_qss[10];
    Kc[129] = -g_RT[7] - g_RT[22] + g_RT_qss[10];
    Kc[130] = -g_RT[3] + g_RT[6] - g_RT[10] - g_RT_qss[6] + g_RT_qss[11];
    Kc[131] = g_RT[4] - g_RT[20] - g_RT_qss[1] + g_RT_qss[11];
    Kc[132] = g_RT[6] - g_RT[17] - g_RT_qss[1] + g_RT_qss[12];
    Kc[133] = -g_RT[6] + g_RT[7] - g_RT[24] + g_RT_qss[12];
    Kc[134] = -g_RT[2] + g_RT[3] - g_RT_qss[11] + g_RT_qss[12];
    Kc[135] = g_RT[4] - g_RT[5] - g_RT_qss[11] + g_RT_qss[12];
    Kc[136] = -g_RT[4] + g_RT[5] + g_RT_qss[11] - g_RT_qss[12];
    Kc[137] = -g_RT[2] + g_RT[3] + g_RT[24] - g_RT_qss[12];
    Kc[138] = g_RT[4] - g_RT[5] + g_RT[24] - g_RT_qss[12];
    Kc[139] = g_RT[1] - g_RT[10] - g_RT[22] + g_RT[24];
    Kc[140] = -g_RT[2] + g_RT[3] - g_RT[24] + g_RT[25];
    Kc[141] = -g_RT[6] + g_RT[7] - g_RT[21] + g_RT[25];
    Kc[142] = g_RT[3] - g_RT[21] + g_RT[25];
    Kc[143] = -g_RT[13] - g_RT[20] + g_RT[25];
    Kc[144] = -g_RT[3] - g_RT[24] + g_RT[25];
    Kc[145] = g_RT[3] + g_RT[24] - g_RT[25];
    Kc[146] = g_RT[11] - g_RT[21] + g_RT[25] - g_RT_qss[1];
    Kc[147] = -g_RT[21] - g_RT[24] + 2.000000*g_RT[25];
    Kc[148] = g_RT[3] - g_RT[13] + g_RT[21] - g_RT[22];
    Kc[149] = -g_RT[2] + g_RT[3] + g_RT[21] - g_RT[25];
    Kc[150] = g_RT[1] + g_RT[21] - g_RT_qss[1] - g_RT_qss[4];
    Kc[151] = g_RT[1] - g_RT[4] + g_RT[21] - g_RT[25];
    Kc[152] = g_RT[1] - g_RT[3] - g_RT[13] - g_RT[17] + g_RT[21];
    Kc[153] = g_RT[4] - g_RT[5] + g_RT[21] - g_RT[25];
    Kc[154] = g_RT[6] - g_RT[7] - g_RT[21] + g_RT_qss[13];
    Kc[155] = -g_RT[13] - g_RT[22] + g_RT_qss[13];
    Kc[156] = g_RT[13] + g_RT[22] - g_RT_qss[13];
    Kc[157] = -g_RT[3] - g_RT[21] + g_RT_qss[13];
    Kc[158] = g_RT[3] + g_RT[21] - g_RT_qss[13];
    Kc[159] = -g_RT[6] + g_RT[26] - g_RT_qss[13];
    Kc[160] = g_RT[6] - g_RT[26] + g_RT_qss[13];
    Kc[161] = g_RT[27] - 2.000000*g_RT_qss[7];
    Kc[162] = -g_RT[27] + 2.000000*g_RT_qss[7];
    Kc[163] = g_RT[4] - g_RT[11] - g_RT[25] + g_RT[27];
    Kc[164] = g_RT[4] - g_RT[17] + g_RT[27] - g_RT_qss[4];
    Kc[165] = g_RT[1] - g_RT[17] - g_RT[22] + g_RT[27];
    Kc[166] = g_RT[3] - g_RT[22] + g_RT[27] - g_RT_qss[7];
    Kc[167] = g_RT[1] - g_RT[11] - g_RT[24] + g_RT[27];
    Kc[168] = g_RT[3] + g_RT[28] - g_RT[29];
    Kc[169] = -g_RT[21] + g_RT[25] - g_RT[27] + g_RT[28];
    Kc[170] = -g_RT[16] - g_RT[27] + g_RT[28] + g_RT_qss[4];
    Kc[171] = -g_RT[3] - g_RT[27] + g_RT[28];
    Kc[172] = g_RT[3] + g_RT[27] - g_RT[28];
    Kc[173] = g_RT[13] - g_RT[14] - g_RT[27] + g_RT[28];
    Kc[174] = -g_RT[6] + g_RT[7] + g_RT[28] - g_RT[29];
    Kc[175] = g_RT[6] - g_RT[7] - g_RT[27] + g_RT[28];
    Kc[176] = -g_RT[22] + g_RT[28] - g_RT_qss[7];
    Kc[177] = -g_RT[2] + g_RT[3] - g_RT[27] + g_RT[28];
    Kc[178] = -g_RT[2] + g_RT[3] - g_RT[28] + g_RT[29];
    Kc[179] = g_RT[4] - g_RT[11] + g_RT[29] - g_RT_qss[13];
    Kc[180] = g_RT[4] - g_RT[16] - g_RT[23] + g_RT[29];
    Kc[181] = g_RT[1] - g_RT[23] + g_RT[29] - g_RT_qss[4];
    Kc[182] = g_RT[1] - g_RT[11] - g_RT[21] + g_RT[29];
    Kc[183] = g_RT[4] - g_RT[5] - g_RT[28] + g_RT[29];
    Kc[184] = -g_RT[13] - g_RT[25] + g_RT[29];
    Kc[185] = g_RT[13] + g_RT[25] - g_RT[29];
    Kc[186] = -g_RT[3] - g_RT[29] + g_RT_qss[14];
    Kc[187] = g_RT[3] + g_RT[29] - g_RT_qss[14];
    Kc[188] = -g_RT[22] - g_RT_qss[4] + g_RT_qss[14];
    Kc[189] = -g_RT[6] - g_RT_qss[14] + g_RT_qss[15];
    Kc[190] = g_RT[6] + g_RT_qss[14] - g_RT_qss[15];
    Kc[191] = -g_RT[13] - g_RT[27] + g_RT_qss[16];
    Kc[192] = -g_RT[22] - g_RT[25] + g_RT_qss[16];
    Kc[193] = g_RT[4] - g_RT[5] - g_RT_qss[16] + g_RT_qss[17];
    Kc[194] = -g_RT[2] + g_RT[3] - g_RT_qss[16] + g_RT_qss[17];
    Kc[195] = -g_RT[25] - g_RT_qss[4] + g_RT_qss[17];
    Kc[196] = g_RT[25] + g_RT_qss[4] - g_RT_qss[17];
    Kc[197] = g_RT[1] - g_RT[4] - g_RT_qss[16] + g_RT_qss[17];
    Kc[198] = -g_RT[21] - g_RT_qss[4] + g_RT_qss[18];
    Kc[199] = -g_RT[22] - g_RT_qss[13] + g_RT_qss[18];
    Kc[200] = -g_RT[3] - g_RT_qss[17] + g_RT_qss[18];
    Kc[201] = -g_RT[25] - g_RT_qss[13] + g_RT_qss[19];
    Kc[202] = g_RT[4] - g_RT[11] - g_RT_qss[18] + g_RT_qss[19];
    Kc[203] = -g_RT[13] + g_RT[30] - g_RT_qss[19];
    Kc[204] = -g_RT[21] + g_RT[30] - g_RT_qss[14];
    Kc[205] = -g_RT[29] + g_RT[30] - g_RT_qss[13];
    Kc[206] = -g_RT[22] + g_RT[30] - g_RT_qss[18];
    Kc[207] = g_RT[30] - g_RT_qss[4] - g_RT_qss[17];
    Kc[208] = -g_RT[6] + g_RT[7] + g_RT[30] - g_RT[31];
    Kc[209] = g_RT[19] - g_RT[30] + g_RT[31] - g_RT_qss[5];
    Kc[210] = -g_RT[2] + g_RT[3] - g_RT[30] + g_RT[31];
    Kc[211] = g_RT[31] - g_RT_qss[13] - g_RT_qss[14];
    Kc[212] = g_RT[7] - g_RT[8] - g_RT[30] + g_RT[31];
    Kc[213] = g_RT[31] - g_RT_qss[4] - g_RT_qss[18];
    Kc[214] = -g_RT[15] - g_RT[30] + g_RT[31] + g_RT_qss[3];
    Kc[215] = g_RT[1] - g_RT[4] - g_RT[30] + g_RT[31];
    Kc[216] = g_RT[4] - g_RT[5] - g_RT[30] + g_RT[31];
    Kc[217] = g_RT[13] - g_RT[14] - g_RT[30] + g_RT[31];

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
    qf[2] = sc[4]*sc[13];
    qr[2] = 0.0;

    /*reaction 4: CH3 + H (+M) => CH4 (+M) */
    qf[3] = sc[3]*sc[13];
    qr[3] = 0.0;

    /*reaction 5: 2.000000 CH3 (+M) => C2H6 (+M) */
    qf[4] = pow(sc[13], 2.000000);
    qr[4] = 0.0;

    /*reaction 6: CO + CH2 (+M) => CH2CO (+M) */
    qf[5] = qss_sc[2]*sc[10];
    qr[5] = 0.0;

    /*reaction 7: CO + O (+M) => CO2 (+M) */
    qf[6] = sc[1]*sc[10];
    qr[6] = 0.0;

    /*reaction 8: CH3O (+M) => CH2O + H (+M) */
    qf[7] = qss_sc[3];
    qr[7] = 0.0;

    /*reaction 9: C2H3 (+M) => H + C2H2 (+M) */
    qf[8] = qss_sc[7];
    qr[8] = 0.0;

    /*reaction 10: H + C2H4 (+M) <=> C2H5 (+M) */
    qf[9] = sc[3]*sc[22];
    qr[9] = qss_sc[4];

    /*reaction 11: CH3CO (+M) => CH3 + CO (+M) */
    qf[10] = sc[23];
    qr[10] = 0.0;

    /*reaction 12: H + OH + M => H2O + M */
    qf[11] = sc[3]*sc[4];
    qr[11] = 0.0;

    /*reaction 13: CH2GSG + M => CH2 + M */
    qf[12] = sc[12];
    qr[12] = 0.0;

    /*reaction 14: CH2 + M => CH2GSG + M */
    qf[13] = qss_sc[2];
    qr[13] = 0.0;

    /*reaction 15: HCO + M => H + CO + M */
    qf[14] = qss_sc[1];
    qr[14] = 0.0;

    /*reaction 16: CH3O2 + M => CH3 + O2 + M */
    qf[15] = sc[19];
    qr[15] = 0.0;

    /*reaction 17: CH3 + O2 + M => CH3O2 + M */
    qf[16] = sc[6]*sc[13];
    qr[16] = 0.0;

    /*reaction 18: C2H5O + M => CH3 + CH2O + M */
    qf[17] = qss_sc[9];
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
    qf[43] = sc[3]*sc[12];
    qr[43] = 0.0;

    /*reaction 45: CH2GSG + H2 => CH3 + H */
    qf[44] = sc[2]*sc[12];
    qr[44] = 0.0;

    /*reaction 46: CH3 + H => CH2GSG + H2 */
    qf[45] = sc[3]*sc[13];
    qr[45] = 0.0;

    /*reaction 47: CH2GSG + O2 => CO + OH + H */
    qf[46] = sc[6]*sc[12];
    qr[46] = 0.0;

    /*reaction 48: CH2GSG + OH => CH2O + H */
    qf[47] = sc[4]*sc[12];
    qr[47] = 0.0;

    /*reaction 49: CH3 + OH => CH2O + H2 */
    qf[48] = sc[4]*sc[13];
    qr[48] = 0.0;

    /*reaction 50: CH3 + OH => CH2GSG + H2O */
    qf[49] = sc[4]*sc[13];
    qr[49] = 0.0;

    /*reaction 51: CH2GSG + H2O => CH3 + OH */
    qf[50] = sc[5]*sc[12];
    qr[50] = 0.0;

    /*reaction 52: CH3 + O => CH2O + H */
    qf[51] = sc[1]*sc[13];
    qr[51] = 0.0;

    /*reaction 53: CH3 + HO2 => CH3O + OH */
    qf[52] = sc[7]*sc[13];
    qr[52] = 0.0;

    /*reaction 54: CH3 + HO2 => CH4 + O2 */
    qf[53] = sc[7]*sc[13];
    qr[53] = 0.0;

    /*reaction 55: CH3 + O2 => CH2O + OH */
    qf[54] = sc[6]*sc[13];
    qr[54] = 0.0;

    /*reaction 56: CH3 + H => CH2 + H2 */
    qf[55] = sc[3]*sc[13];
    qr[55] = 0.0;

    /*reaction 57: CH2 + H2 => CH3 + H */
    qf[56] = sc[2]*qss_sc[2];
    qr[56] = 0.0;

    /*reaction 58: 2.000000 CH3 <=> H + C2H5 */
    qf[57] = pow(sc[13], 2.000000);
    qr[57] = sc[3]*qss_sc[4];

    /*reaction 59: CH3 + OH => CH2 + H2O */
    qf[58] = sc[4]*sc[13];
    qr[58] = 0.0;

    /*reaction 60: CH2 + H2O => CH3 + OH */
    qf[59] = sc[5]*qss_sc[2];
    qr[59] = 0.0;

    /*reaction 61: CH4 + O => CH3 + OH */
    qf[60] = sc[1]*sc[14];
    qr[60] = 0.0;

    /*reaction 62: CH4 + H => CH3 + H2 */
    qf[61] = sc[3]*sc[14];
    qr[61] = 0.0;

    /*reaction 63: CH3 + H2 => CH4 + H */
    qf[62] = sc[2]*sc[13];
    qr[62] = 0.0;

    /*reaction 64: CH4 + OH => CH3 + H2O */
    qf[63] = sc[4]*sc[14];
    qr[63] = 0.0;

    /*reaction 65: CH3 + H2O => CH4 + OH */
    qf[64] = sc[5]*sc[13];
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
    qf[72] = qss_sc[1]*sc[13];
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
    qf[76] = sc[11]*sc[13];
    qr[76] = 0.0;

    /*reaction 78: 2.000000 CH3O => CH3OH + CH2O */
    qf[77] = pow(qss_sc[3], 2.000000);
    qr[77] = 0.0;

    /*reaction 79: CH3O + O2 => CH2O + HO2 */
    qf[78] = sc[6]*qss_sc[3];
    qr[78] = 0.0;

    /*reaction 80: CH3O + H2 => CH3OH + H */
    qf[79] = sc[2]*qss_sc[3];
    qr[79] = 0.0;

    /*reaction 81: CH3OH + OH => CH3O + H2O */
    qf[80] = sc[4]*sc[15];
    qr[80] = 0.0;

    /*reaction 82: CH2GSG + CO2 => CH2O + CO */
    qf[81] = sc[9]*sc[12];
    qr[81] = 0.0;

    /*reaction 83: HOCHO + H => H2 + CO + OH */
    qf[82] = sc[3]*sc[18];
    qr[82] = 0.0;

    /*reaction 84: HOCHO + OH => H2O + CO + OH */
    qf[83] = sc[4]*sc[18];
    qr[83] = 0.0;

    /*reaction 85: HOCHO => HCO + OH */
    qf[84] = sc[18];
    qr[84] = 0.0;

    /*reaction 86: HCO + OH => HOCHO */
    qf[85] = sc[4]*qss_sc[1];
    qr[85] = 0.0;

    /*reaction 87: HOCHO + H => H2 + CO2 + H */
    qf[86] = sc[3]*sc[18];
    qr[86] = 0.0;

    /*reaction 88: HOCHO + OH => H2O + CO2 + H */
    qf[87] = sc[4]*sc[18];
    qr[87] = 0.0;

    /*reaction 89: 2.000000 CH3O2 => O2 + 2.000000 CH3O */
    qf[88] = pow(sc[19], 2.000000);
    qr[88] = 0.0;

    /*reaction 90: CH3O2 + CH3 => 2.000000 CH3O */
    qf[89] = sc[13]*sc[19];
    qr[89] = 0.0;

    /*reaction 91: 2.000000 CH3O2 => CH2O + CH3OH + O2 */
    qf[90] = pow(sc[19], 2.000000);
    qr[90] = 0.0;

    /*reaction 92: CH3O2 + HO2 => CH3O2H + O2 */
    qf[91] = sc[7]*sc[19];
    qr[91] = 0.0;

    /*reaction 93: CH3O2H => CH3O + OH */
    qf[92] = qss_sc[5];
    qr[92] = 0.0;

    /*reaction 94: C2H2 + O => CH2 + CO */
    qf[93] = sc[1]*sc[20];
    qr[93] = 0.0;

    /*reaction 95: C2H2 + O => HCCO + H */
    qf[94] = sc[1]*sc[20];
    qr[94] = 0.0;

    /*reaction 96: C2H3 + H => C2H2 + H2 */
    qf[95] = sc[3]*qss_sc[7];
    qr[95] = 0.0;

    /*reaction 97: C2H3 + O2 => CH2CHO + O */
    qf[96] = sc[6]*qss_sc[7];
    qr[96] = 0.0;

    /*reaction 98: C2H3 + CH3 => C3H6 */
    qf[97] = sc[13]*qss_sc[7];
    qr[97] = 0.0;

    /*reaction 99: C2H3 + O2 => C2H2 + HO2 */
    qf[98] = sc[6]*qss_sc[7];
    qr[98] = 0.0;

    /*reaction 100: C2H3 + O2 => CH2O + HCO */
    qf[99] = sc[6]*qss_sc[7];
    qr[99] = 0.0;

    /*reaction 101: C2H4 + CH3 => C2H3 + CH4 */
    qf[100] = sc[13]*sc[22];
    qr[100] = 0.0;

    /*reaction 102: C2H4 + O => CH3 + HCO */
    qf[101] = sc[1]*sc[22];
    qr[101] = 0.0;

    /*reaction 103: C2H4 + OH => C2H3 + H2O */
    qf[102] = sc[4]*sc[22];
    qr[102] = 0.0;

    /*reaction 104: C2H4 + O => CH2CHO + H */
    qf[103] = sc[1]*sc[22];
    qr[103] = 0.0;

    /*reaction 105: C2H4 + H => C2H3 + H2 */
    qf[104] = sc[3]*sc[22];
    qr[104] = 0.0;

    /*reaction 106: C2H3 + H2 => C2H4 + H */
    qf[105] = sc[2]*qss_sc[7];
    qr[105] = 0.0;

    /*reaction 107: H + C2H5 => C2H6 */
    qf[106] = sc[3]*qss_sc[4];
    qr[106] = 0.0;

    /*reaction 108: CH3O2 + C2H5 => CH3O + C2H5O */
    qf[107] = qss_sc[4]*sc[19];
    qr[107] = 0.0;

    /*reaction 109: C2H5 + HO2 => C2H5O + OH */
    qf[108] = sc[7]*qss_sc[4];
    qr[108] = 0.0;

    /*reaction 110: C2H5 + O2 => C2H4 + HO2 */
    qf[109] = sc[6]*qss_sc[4];
    qr[109] = 0.0;

    /*reaction 111: C2H6 + O => C2H5 + OH */
    qf[110] = sc[1]*sc[16];
    qr[110] = 0.0;

    /*reaction 112: C2H6 + OH => C2H5 + H2O */
    qf[111] = sc[4]*sc[16];
    qr[111] = 0.0;

    /*reaction 113: C2H6 + H => C2H5 + H2 */
    qf[112] = sc[3]*sc[16];
    qr[112] = 0.0;

    /*reaction 114: HCCO + O => H + 2.000000 CO */
    qf[113] = sc[1]*qss_sc[6];
    qr[113] = 0.0;

    /*reaction 115: HCCO + OH => 2.000000 HCO */
    qf[114] = sc[4]*qss_sc[6];
    qr[114] = 0.0;

    /*reaction 116: HCCO + O2 => CO2 + HCO */
    qf[115] = sc[6]*qss_sc[6];
    qr[115] = 0.0;

    /*reaction 117: HCCO + H => CH2GSG + CO */
    qf[116] = sc[3]*qss_sc[6];
    qr[116] = 0.0;

    /*reaction 118: CH2GSG + CO => HCCO + H */
    qf[117] = sc[10]*sc[12];
    qr[117] = 0.0;

    /*reaction 119: CH2CO + O => HCCO + OH */
    qf[118] = sc[1]*sc[17];
    qr[118] = 0.0;

    /*reaction 120: CH2CO + H => HCCO + H2 */
    qf[119] = sc[3]*sc[17];
    qr[119] = 0.0;

    /*reaction 121: HCCO + H2 => CH2CO + H */
    qf[120] = sc[2]*qss_sc[6];
    qr[120] = 0.0;

    /*reaction 122: CH2CO + H => CH3 + CO */
    qf[121] = sc[3]*sc[17];
    qr[121] = 0.0;

    /*reaction 123: CH2CO + O => CH2 + CO2 */
    qf[122] = sc[1]*sc[17];
    qr[122] = 0.0;

    /*reaction 124: CH2CO + OH => HCCO + H2O */
    qf[123] = sc[4]*sc[17];
    qr[123] = 0.0;

    /*reaction 125: CH2CHO + O2 => CH2O + CO + OH */
    qf[124] = sc[6]*qss_sc[8];
    qr[124] = 0.0;

    /*reaction 126: CH2CHO => CH2CO + H */
    qf[125] = qss_sc[8];
    qr[125] = 0.0;

    /*reaction 127: CH2CO + H => CH2CHO */
    qf[126] = sc[3]*sc[17];
    qr[126] = 0.0;

    /*reaction 128: C2H5O2 => C2H5 + O2 */
    qf[127] = qss_sc[10];
    qr[127] = 0.0;

    /*reaction 129: C2H5 + O2 => C2H5O2 */
    qf[128] = sc[6]*qss_sc[4];
    qr[128] = 0.0;

    /*reaction 130: C2H5O2 => C2H4 + HO2 */
    qf[129] = qss_sc[10];
    qr[129] = 0.0;

    /*reaction 131: C3H2 + O2 => HCCO + CO + H */
    qf[130] = sc[6]*qss_sc[11];
    qr[130] = 0.0;

    /*reaction 132: C3H2 + OH => C2H2 + HCO */
    qf[131] = sc[4]*qss_sc[11];
    qr[131] = 0.0;

    /*reaction 133: C3H3 + O2 => CH2CO + HCO */
    qf[132] = sc[6]*qss_sc[12];
    qr[132] = 0.0;

    /*reaction 134: C3H3 + HO2 => C3H4XA + O2 */
    qf[133] = sc[7]*qss_sc[12];
    qr[133] = 0.0;

    /*reaction 135: C3H3 + H => C3H2 + H2 */
    qf[134] = sc[3]*qss_sc[12];
    qr[134] = 0.0;

    /*reaction 136: C3H3 + OH => C3H2 + H2O */
    qf[135] = sc[4]*qss_sc[12];
    qr[135] = 0.0;

    /*reaction 137: C3H2 + H2O => C3H3 + OH */
    qf[136] = sc[5]*qss_sc[11];
    qr[136] = 0.0;

    /*reaction 138: C3H4XA + H => C3H3 + H2 */
    qf[137] = sc[3]*sc[24];
    qr[137] = 0.0;

    /*reaction 139: C3H4XA + OH => C3H3 + H2O */
    qf[138] = sc[4]*sc[24];
    qr[138] = 0.0;

    /*reaction 140: C3H4XA + O => C2H4 + CO */
    qf[139] = sc[1]*sc[24];
    qr[139] = 0.0;

    /*reaction 141: C3H5XA + H => C3H4XA + H2 */
    qf[140] = sc[3]*sc[25];
    qr[140] = 0.0;

    /*reaction 142: C3H5XA + HO2 => C3H6 + O2 */
    qf[141] = sc[7]*sc[25];
    qr[141] = 0.0;

    /*reaction 143: C3H5XA + H => C3H6 */
    qf[142] = sc[3]*sc[25];
    qr[142] = 0.0;

    /*reaction 144: C3H5XA => C2H2 + CH3 */
    qf[143] = sc[25];
    qr[143] = 0.0;

    /*reaction 145: C3H5XA => C3H4XA + H */
    qf[144] = sc[25];
    qr[144] = 0.0;

    /*reaction 146: C3H4XA + H => C3H5XA */
    qf[145] = sc[3]*sc[24];
    qr[145] = 0.0;

    /*reaction 147: C3H5XA + CH2O => C3H6 + HCO */
    qf[146] = sc[11]*sc[25];
    qr[146] = 0.0;

    /*reaction 148: 2.000000 C3H5XA => C3H4XA + C3H6 */
    qf[147] = pow(sc[25], 2.000000);
    qr[147] = 0.0;

    /*reaction 149: C3H6 + H => C2H4 + CH3 */
    qf[148] = sc[3]*sc[21];
    qr[148] = 0.0;

    /*reaction 150: C3H6 + H => C3H5XA + H2 */
    qf[149] = sc[3]*sc[21];
    qr[149] = 0.0;

    /*reaction 151: C3H6 + O => C2H5 + HCO */
    qf[150] = sc[1]*sc[21];
    qr[150] = 0.0;

    /*reaction 152: C3H6 + O => C3H5XA + OH */
    qf[151] = sc[1]*sc[21];
    qr[151] = 0.0;

    /*reaction 153: C3H6 + O => CH2CO + CH3 + H */
    qf[152] = sc[1]*sc[21];
    qr[152] = 0.0;

    /*reaction 154: C3H6 + OH => C3H5XA + H2O */
    qf[153] = sc[4]*sc[21];
    qr[153] = 0.0;

    /*reaction 155: NXC3H7 + O2 => C3H6 + HO2 */
    qf[154] = sc[6]*qss_sc[13];
    qr[154] = 0.0;

    /*reaction 156: NXC3H7 => CH3 + C2H4 */
    qf[155] = qss_sc[13];
    qr[155] = 0.0;

    /*reaction 157: CH3 + C2H4 => NXC3H7 */
    qf[156] = sc[13]*sc[22];
    qr[156] = 0.0;

    /*reaction 158: NXC3H7 => H + C3H6 */
    qf[157] = qss_sc[13];
    qr[157] = 0.0;

    /*reaction 159: H + C3H6 => NXC3H7 */
    qf[158] = sc[3]*sc[21];
    qr[158] = 0.0;

    /*reaction 160: NXC3H7O2 => NXC3H7 + O2 */
    qf[159] = sc[26];
    qr[159] = 0.0;

    /*reaction 161: NXC3H7 + O2 => NXC3H7O2 */
    qf[160] = sc[6]*qss_sc[13];
    qr[160] = 0.0;

    /*reaction 162: C4H6 => 2.000000 C2H3 */
    qf[161] = sc[27];
    qr[161] = 0.0;

    /*reaction 163: 2.000000 C2H3 => C4H6 */
    qf[162] = pow(qss_sc[7], 2.000000);
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
    qf[169] = sc[25]*sc[28];
    qr[169] = 0.0;

    /*reaction 171: C2H5 + C4H7 => C4H6 + C2H6 */
    qf[170] = qss_sc[4]*sc[28];
    qr[170] = 0.0;

    /*reaction 172: C4H7 => C4H6 + H */
    qf[171] = sc[28];
    qr[171] = 0.0;

    /*reaction 173: C4H6 + H => C4H7 */
    qf[172] = sc[3]*sc[27];
    qr[172] = 0.0;

    /*reaction 174: C4H7 + CH3 => C4H6 + CH4 */
    qf[173] = sc[13]*sc[28];
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
    qf[185] = sc[13]*sc[25];
    qr[185] = 0.0;

    /*reaction 187: PXC4H9 => C4H8X1 + H */
    qf[186] = qss_sc[14];
    qr[186] = 0.0;

    /*reaction 188: C4H8X1 + H => PXC4H9 */
    qf[187] = sc[3]*sc[29];
    qr[187] = 0.0;

    /*reaction 189: PXC4H9 => C2H5 + C2H4 */
    qf[188] = qss_sc[14];
    qr[188] = 0.0;

    /*reaction 190: PXC4H9O2 => PXC4H9 + O2 */
    qf[189] = qss_sc[15];
    qr[189] = 0.0;

    /*reaction 191: PXC4H9 + O2 => PXC4H9O2 */
    qf[190] = sc[6]*qss_sc[14];
    qr[190] = 0.0;

    /*reaction 192: C5H9 => C4H6 + CH3 */
    qf[191] = qss_sc[16];
    qr[191] = 0.0;

    /*reaction 193: C5H9 => C3H5XA + C2H4 */
    qf[192] = qss_sc[16];
    qr[192] = 0.0;

    /*reaction 194: C5H10X1 + OH => C5H9 + H2O */
    qf[193] = sc[4]*qss_sc[17];
    qr[193] = 0.0;

    /*reaction 195: C5H10X1 + H => C5H9 + H2 */
    qf[194] = sc[3]*qss_sc[17];
    qr[194] = 0.0;

    /*reaction 196: C5H10X1 => C2H5 + C3H5XA */
    qf[195] = qss_sc[17];
    qr[195] = 0.0;

    /*reaction 197: C2H5 + C3H5XA => C5H10X1 */
    qf[196] = qss_sc[4]*sc[25];
    qr[196] = 0.0;

    /*reaction 198: C5H10X1 + O => C5H9 + OH */
    qf[197] = sc[1]*qss_sc[17];
    qr[197] = 0.0;

    /*reaction 199: C5H11X1 => C3H6 + C2H5 */
    qf[198] = qss_sc[18];
    qr[198] = 0.0;

    /*reaction 200: C5H11X1 => C2H4 + NXC3H7 */
    qf[199] = qss_sc[18];
    qr[199] = 0.0;

    /*reaction 201: C5H11X1 <=> C5H10X1 + H */
    qf[200] = qss_sc[18];
    qr[200] = sc[3]*qss_sc[17];

    /*reaction 202: C6H12X1 => NXC3H7 + C3H5XA */
    qf[201] = qss_sc[19];
    qr[201] = 0.0;

    /*reaction 203: C6H12X1 + OH => C5H11X1 + CH2O */
    qf[202] = sc[4]*qss_sc[19];
    qr[202] = 0.0;

    /*reaction 204: C7H15X2 => C6H12X1 + CH3 */
    qf[203] = sc[30];
    qr[203] = 0.0;

    /*reaction 205: C7H15X2 => PXC4H9 + C3H6 */
    qf[204] = sc[30];
    qr[204] = 0.0;

    /*reaction 206: C7H15X2 => C4H8X1 + NXC3H7 */
    qf[205] = sc[30];
    qr[205] = 0.0;

    /*reaction 207: C7H15X2 => C5H11X1 + C2H4 */
    qf[206] = sc[30];
    qr[206] = 0.0;

    /*reaction 208: C7H15X2 => C2H5 + C5H10X1 */
    qf[207] = sc[30];
    qr[207] = 0.0;

    /*reaction 209: C7H15X2 + HO2 => NXC7H16 + O2 */
    qf[208] = sc[7]*sc[30];
    qr[208] = 0.0;

    /*reaction 210: NXC7H16 + CH3O2 => C7H15X2 + CH3O2H */
    qf[209] = sc[19]*sc[31];
    qr[209] = 0.0;

    /*reaction 211: NXC7H16 + H => C7H15X2 + H2 */
    qf[210] = sc[3]*sc[31];
    qr[210] = 0.0;

    /*reaction 212: NXC7H16 => PXC4H9 + NXC3H7 */
    qf[211] = sc[31];
    qr[211] = 0.0;

    /*reaction 213: NXC7H16 + HO2 => C7H15X2 + H2O2 */
    qf[212] = sc[7]*sc[31];
    qr[212] = 0.0;

    /*reaction 214: NXC7H16 => C5H11X1 + C2H5 */
    qf[213] = sc[31];
    qr[213] = 0.0;

    /*reaction 215: NXC7H16 + CH3O => C7H15X2 + CH3OH */
    qf[214] = qss_sc[3]*sc[31];
    qr[214] = 0.0;

    /*reaction 216: NXC7H16 + O => C7H15X2 + OH */
    qf[215] = sc[1]*sc[31];
    qr[215] = 0.0;

    /*reaction 217: NXC7H16 + OH => C7H15X2 + H2O */
    qf[216] = sc[4]*sc[31];
    qr[216] = 0.0;

    /*reaction 218: NXC7H16 + CH3 => C7H15X2 + CH4 */
    qf[217] = sc[13]*sc[31];
    qr[217] = 0.0;

    double T = tc[1];

    /*compute the mixture concentration */
    double mixture = 0.0;
    for (int i = 0; i < 32; ++i) {
        mixture += sc[i];
    }

    double Corr[218];
    for (int i = 0; i < 218; ++i) {
        Corr[i] = 1.0;
    }

    /* troe */
    {
        double alpha[11];
        alpha[0] = mixture + (TB[0][2] - 1)*sc[10] + (TB[0][3] - 1)*sc[2] + (TB[0][5] - 1)*sc[12] + (TB[0][6] - 1)*sc[28] + (TB[0][9] - 1)*sc[5] + (TB[0][10] - 1)*sc[9] + (TB[0][13] - 1)*sc[23] + (TB[0][14] - 1)*sc[30];
        alpha[1] = mixture + (TB[1][2] - 1)*sc[10] + (TB[1][3] - 1)*sc[2] + (TB[1][5] - 1)*sc[12] + (TB[1][6] - 1)*sc[28] + (TB[1][9] - 1)*sc[5] + (TB[1][10] - 1)*sc[9] + (TB[1][13] - 1)*sc[23] + (TB[1][14] - 1)*sc[30];
        alpha[2] = mixture + (TB[2][2] - 1)*sc[10] + (TB[2][3] - 1)*sc[2] + (TB[2][5] - 1)*sc[12] + (TB[2][6] - 1)*sc[28] + (TB[2][9] - 1)*sc[5] + (TB[2][10] - 1)*sc[9] + (TB[2][11] - 1)*sc[14] + (TB[2][12] - 1)*sc[16] + (TB[2][15] - 1)*sc[23] + (TB[2][16] - 1)*sc[30];
        alpha[3] = mixture + (TB[3][2] - 1)*sc[10] + (TB[3][3] - 1)*sc[2] + (TB[3][5] - 1)*sc[12] + (TB[3][6] - 1)*sc[28] + (TB[3][9] - 1)*sc[5] + (TB[3][10] - 1)*sc[9] + (TB[3][13] - 1)*sc[23] + (TB[3][14] - 1)*sc[30];
        alpha[4] = mixture + (TB[4][2] - 1)*sc[10] + (TB[4][3] - 1)*sc[2] + (TB[4][5] - 1)*sc[12] + (TB[4][6] - 1)*sc[28] + (TB[4][9] - 1)*sc[5] + (TB[4][10] - 1)*sc[9] + (TB[4][13] - 1)*sc[23] + (TB[4][14] - 1)*sc[30];
        alpha[5] = mixture + (TB[5][4] - 1)*sc[12] + (TB[5][5] - 1)*sc[28] + (TB[5][8] - 1)*sc[23] + (TB[5][10] - 1)*sc[30];
        alpha[6] = mixture + (TB[6][2] - 1)*sc[10] + (TB[6][3] - 1)*sc[2] + (TB[6][5] - 1)*sc[12] + (TB[6][6] - 1)*sc[28] + (TB[6][9] - 1)*sc[5] + (TB[6][10] - 1)*sc[9] + (TB[6][13] - 1)*sc[23] + (TB[6][14] - 1)*sc[30];
        alpha[7] = mixture + (TB[7][4] - 1)*sc[12] + (TB[7][5] - 1)*sc[28] + (TB[7][8] - 1)*sc[23] + (TB[7][10] - 1)*sc[30];
        alpha[8] = mixture + (TB[8][2] - 1)*sc[10] + (TB[8][3] - 1)*sc[2] + (TB[8][5] - 1)*sc[12] + (TB[8][6] - 1)*sc[28] + (TB[8][9] - 1)*sc[5] + (TB[8][10] - 1)*sc[9] + (TB[8][13] - 1)*sc[23] + (TB[8][14] - 1)*sc[30];
        alpha[9] = mixture + (TB[9][4] - 1)*sc[12] + (TB[9][5] - 1)*sc[28] + (TB[9][8] - 1)*sc[23] + (TB[9][10] - 1)*sc[30];
        alpha[10] = mixture + (TB[10][4] - 1)*sc[12] + (TB[10][5] - 1)*sc[28] + (TB[10][8] - 1)*sc[23] + (TB[10][10] - 1)*sc[30];
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
        alpha = mixture + (TB[11][2] - 1)*sc[10] + (TB[11][3] - 1)*sc[2] + (TB[11][5] - 1)*sc[12] + (TB[11][6] - 1)*sc[28] + (TB[11][9] - 1)*sc[5] + (TB[11][10] - 1)*sc[9] + (TB[11][13] - 1)*sc[23] + (TB[11][14] - 1)*sc[30];
        Corr[11] = alpha;
        alpha = mixture + (TB[12][4] - 1)*sc[12] + (TB[12][5] - 1)*sc[28] + (TB[12][8] - 1)*sc[23] + (TB[12][10] - 1)*sc[30];
        Corr[12] = alpha;
        alpha = mixture + (TB[13][4] - 1)*sc[12] + (TB[13][5] - 1)*sc[28] + (TB[13][8] - 1)*sc[23] + (TB[13][10] - 1)*sc[30];
        Corr[13] = alpha;
        alpha = mixture + (TB[14][2] - 1)*sc[10] + (TB[14][3] - 1)*sc[2] + (TB[14][5] - 1)*sc[12] + (TB[14][6] - 1)*sc[28] + (TB[14][9] - 1)*sc[5] + (TB[14][10] - 1)*sc[9] + (TB[14][13] - 1)*sc[23] + (TB[14][14] - 1)*sc[30];
        Corr[14] = alpha;
        alpha = mixture + (TB[15][4] - 1)*sc[12] + (TB[15][5] - 1)*sc[28] + (TB[15][8] - 1)*sc[23] + (TB[15][10] - 1)*sc[30];
        Corr[15] = alpha;
        alpha = mixture + (TB[16][4] - 1)*sc[12] + (TB[16][5] - 1)*sc[28] + (TB[16][8] - 1)*sc[23] + (TB[16][10] - 1)*sc[30];
        Corr[16] = alpha;
        alpha = mixture + (TB[17][4] - 1)*sc[12] + (TB[17][5] - 1)*sc[28] + (TB[17][8] - 1)*sc[23] + (TB[17][10] - 1)*sc[30];
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
    k_f[4] = prefactor_units[12] * fwd_A[12]
               * exp(fwd_beta[12] * tc[0] - activation_units[12] * fwd_Ea[12] * invT);
    k_f[5] = prefactor_units[13] * fwd_A[13]
               * exp(fwd_beta[13] * tc[0] - activation_units[13] * fwd_Ea[13] * invT);
    k_f[6] = prefactor_units[14] * fwd_A[14]
               * exp(fwd_beta[14] * tc[0] - activation_units[14] * fwd_Ea[14] * invT);
    k_f[7] = prefactor_units[17] * fwd_A[17]
               * exp(fwd_beta[17] * tc[0] - activation_units[17] * fwd_Ea[17] * invT);
    k_f[8] = prefactor_units[33] * fwd_A[33]
               * exp(fwd_beta[33] * tc[0] - activation_units[33] * fwd_Ea[33] * invT);
    k_f[9] = prefactor_units[34] * fwd_A[34]
               * exp(fwd_beta[34] * tc[0] - activation_units[34] * fwd_Ea[34] * invT);
    k_f[10] = prefactor_units[35] * fwd_A[35]
               * exp(fwd_beta[35] * tc[0] - activation_units[35] * fwd_Ea[35] * invT);
    k_f[11] = prefactor_units[36] * fwd_A[36]
               * exp(fwd_beta[36] * tc[0] - activation_units[36] * fwd_Ea[36] * invT);
    k_f[12] = prefactor_units[37] * fwd_A[37]
               * exp(fwd_beta[37] * tc[0] - activation_units[37] * fwd_Ea[37] * invT);
    k_f[13] = prefactor_units[38] * fwd_A[38]
               * exp(fwd_beta[38] * tc[0] - activation_units[38] * fwd_Ea[38] * invT);
    k_f[14] = prefactor_units[39] * fwd_A[39]
               * exp(fwd_beta[39] * tc[0] - activation_units[39] * fwd_Ea[39] * invT);
    k_f[15] = prefactor_units[40] * fwd_A[40]
               * exp(fwd_beta[40] * tc[0] - activation_units[40] * fwd_Ea[40] * invT);
    k_f[16] = prefactor_units[41] * fwd_A[41]
               * exp(fwd_beta[41] * tc[0] - activation_units[41] * fwd_Ea[41] * invT);
    k_f[17] = prefactor_units[42] * fwd_A[42]
               * exp(fwd_beta[42] * tc[0] - activation_units[42] * fwd_Ea[42] * invT);
    k_f[18] = prefactor_units[43] * fwd_A[43]
               * exp(fwd_beta[43] * tc[0] - activation_units[43] * fwd_Ea[43] * invT);
    k_f[19] = prefactor_units[52] * fwd_A[52]
               * exp(fwd_beta[52] * tc[0] - activation_units[52] * fwd_Ea[52] * invT);
    k_f[20] = prefactor_units[55] * fwd_A[55]
               * exp(fwd_beta[55] * tc[0] - activation_units[55] * fwd_Ea[55] * invT);
    k_f[21] = prefactor_units[56] * fwd_A[56]
               * exp(fwd_beta[56] * tc[0] - activation_units[56] * fwd_Ea[56] * invT);
    k_f[22] = prefactor_units[57] * fwd_A[57]
               * exp(fwd_beta[57] * tc[0] - activation_units[57] * fwd_Ea[57] * invT);
    k_f[23] = prefactor_units[58] * fwd_A[58]
               * exp(fwd_beta[58] * tc[0] - activation_units[58] * fwd_Ea[58] * invT);
    k_f[24] = prefactor_units[59] * fwd_A[59]
               * exp(fwd_beta[59] * tc[0] - activation_units[59] * fwd_Ea[59] * invT);
    k_f[25] = prefactor_units[67] * fwd_A[67]
               * exp(fwd_beta[67] * tc[0] - activation_units[67] * fwd_Ea[67] * invT);
    k_f[26] = prefactor_units[68] * fwd_A[68]
               * exp(fwd_beta[68] * tc[0] - activation_units[68] * fwd_Ea[68] * invT);
    k_f[27] = prefactor_units[69] * fwd_A[69]
               * exp(fwd_beta[69] * tc[0] - activation_units[69] * fwd_Ea[69] * invT);
    k_f[28] = prefactor_units[70] * fwd_A[70]
               * exp(fwd_beta[70] * tc[0] - activation_units[70] * fwd_Ea[70] * invT);
    k_f[29] = prefactor_units[71] * fwd_A[71]
               * exp(fwd_beta[71] * tc[0] - activation_units[71] * fwd_Ea[71] * invT);
    k_f[30] = prefactor_units[72] * fwd_A[72]
               * exp(fwd_beta[72] * tc[0] - activation_units[72] * fwd_Ea[72] * invT);
    k_f[31] = prefactor_units[73] * fwd_A[73]
               * exp(fwd_beta[73] * tc[0] - activation_units[73] * fwd_Ea[73] * invT);
    k_f[32] = prefactor_units[74] * fwd_A[74]
               * exp(fwd_beta[74] * tc[0] - activation_units[74] * fwd_Ea[74] * invT);
    k_f[33] = prefactor_units[75] * fwd_A[75]
               * exp(fwd_beta[75] * tc[0] - activation_units[75] * fwd_Ea[75] * invT);
    k_f[34] = prefactor_units[76] * fwd_A[76]
               * exp(fwd_beta[76] * tc[0] - activation_units[76] * fwd_Ea[76] * invT);
    k_f[35] = prefactor_units[77] * fwd_A[77]
               * exp(fwd_beta[77] * tc[0] - activation_units[77] * fwd_Ea[77] * invT);
    k_f[36] = prefactor_units[78] * fwd_A[78]
               * exp(fwd_beta[78] * tc[0] - activation_units[78] * fwd_Ea[78] * invT);
    k_f[37] = prefactor_units[79] * fwd_A[79]
               * exp(fwd_beta[79] * tc[0] - activation_units[79] * fwd_Ea[79] * invT);
    k_f[38] = prefactor_units[80] * fwd_A[80]
               * exp(fwd_beta[80] * tc[0] - activation_units[80] * fwd_Ea[80] * invT);
    k_f[39] = prefactor_units[84] * fwd_A[84]
               * exp(fwd_beta[84] * tc[0] - activation_units[84] * fwd_Ea[84] * invT);
    k_f[40] = prefactor_units[85] * fwd_A[85]
               * exp(fwd_beta[85] * tc[0] - activation_units[85] * fwd_Ea[85] * invT);
    k_f[41] = prefactor_units[88] * fwd_A[88]
               * exp(fwd_beta[88] * tc[0] - activation_units[88] * fwd_Ea[88] * invT);
    k_f[42] = prefactor_units[89] * fwd_A[89]
               * exp(fwd_beta[89] * tc[0] - activation_units[89] * fwd_Ea[89] * invT);
    k_f[43] = prefactor_units[91] * fwd_A[91]
               * exp(fwd_beta[91] * tc[0] - activation_units[91] * fwd_Ea[91] * invT);
    k_f[44] = prefactor_units[92] * fwd_A[92]
               * exp(fwd_beta[92] * tc[0] - activation_units[92] * fwd_Ea[92] * invT);
    k_f[45] = prefactor_units[93] * fwd_A[93]
               * exp(fwd_beta[93] * tc[0] - activation_units[93] * fwd_Ea[93] * invT);
    k_f[46] = prefactor_units[94] * fwd_A[94]
               * exp(fwd_beta[94] * tc[0] - activation_units[94] * fwd_Ea[94] * invT);
    k_f[47] = prefactor_units[95] * fwd_A[95]
               * exp(fwd_beta[95] * tc[0] - activation_units[95] * fwd_Ea[95] * invT);
    k_f[48] = prefactor_units[96] * fwd_A[96]
               * exp(fwd_beta[96] * tc[0] - activation_units[96] * fwd_Ea[96] * invT);
    k_f[49] = prefactor_units[97] * fwd_A[97]
               * exp(fwd_beta[97] * tc[0] - activation_units[97] * fwd_Ea[97] * invT);
    k_f[50] = prefactor_units[98] * fwd_A[98]
               * exp(fwd_beta[98] * tc[0] - activation_units[98] * fwd_Ea[98] * invT);
    k_f[51] = prefactor_units[99] * fwd_A[99]
               * exp(fwd_beta[99] * tc[0] - activation_units[99] * fwd_Ea[99] * invT);
    k_f[52] = prefactor_units[100] * fwd_A[100]
               * exp(fwd_beta[100] * tc[0] - activation_units[100] * fwd_Ea[100] * invT);
    k_f[53] = prefactor_units[101] * fwd_A[101]
               * exp(fwd_beta[101] * tc[0] - activation_units[101] * fwd_Ea[101] * invT);
    k_f[54] = prefactor_units[102] * fwd_A[102]
               * exp(fwd_beta[102] * tc[0] - activation_units[102] * fwd_Ea[102] * invT);
    k_f[55] = prefactor_units[103] * fwd_A[103]
               * exp(fwd_beta[103] * tc[0] - activation_units[103] * fwd_Ea[103] * invT);
    k_f[56] = prefactor_units[104] * fwd_A[104]
               * exp(fwd_beta[104] * tc[0] - activation_units[104] * fwd_Ea[104] * invT);
    k_f[57] = prefactor_units[105] * fwd_A[105]
               * exp(fwd_beta[105] * tc[0] - activation_units[105] * fwd_Ea[105] * invT);
    k_f[58] = prefactor_units[106] * fwd_A[106]
               * exp(fwd_beta[106] * tc[0] - activation_units[106] * fwd_Ea[106] * invT);
    k_f[59] = prefactor_units[107] * fwd_A[107]
               * exp(fwd_beta[107] * tc[0] - activation_units[107] * fwd_Ea[107] * invT);
    k_f[60] = prefactor_units[108] * fwd_A[108]
               * exp(fwd_beta[108] * tc[0] - activation_units[108] * fwd_Ea[108] * invT);
    k_f[61] = prefactor_units[109] * fwd_A[109]
               * exp(fwd_beta[109] * tc[0] - activation_units[109] * fwd_Ea[109] * invT);
    k_f[62] = prefactor_units[110] * fwd_A[110]
               * exp(fwd_beta[110] * tc[0] - activation_units[110] * fwd_Ea[110] * invT);
    k_f[63] = prefactor_units[111] * fwd_A[111]
               * exp(fwd_beta[111] * tc[0] - activation_units[111] * fwd_Ea[111] * invT);
    k_f[64] = prefactor_units[112] * fwd_A[112]
               * exp(fwd_beta[112] * tc[0] - activation_units[112] * fwd_Ea[112] * invT);
    k_f[65] = prefactor_units[113] * fwd_A[113]
               * exp(fwd_beta[113] * tc[0] - activation_units[113] * fwd_Ea[113] * invT);
    k_f[66] = prefactor_units[114] * fwd_A[114]
               * exp(fwd_beta[114] * tc[0] - activation_units[114] * fwd_Ea[114] * invT);
    k_f[67] = prefactor_units[115] * fwd_A[115]
               * exp(fwd_beta[115] * tc[0] - activation_units[115] * fwd_Ea[115] * invT);
    k_f[68] = prefactor_units[116] * fwd_A[116]
               * exp(fwd_beta[116] * tc[0] - activation_units[116] * fwd_Ea[116] * invT);
    k_f[69] = prefactor_units[117] * fwd_A[117]
               * exp(fwd_beta[117] * tc[0] - activation_units[117] * fwd_Ea[117] * invT);
    k_f[70] = prefactor_units[118] * fwd_A[118]
               * exp(fwd_beta[118] * tc[0] - activation_units[118] * fwd_Ea[118] * invT);
    k_f[71] = prefactor_units[119] * fwd_A[119]
               * exp(fwd_beta[119] * tc[0] - activation_units[119] * fwd_Ea[119] * invT);
    k_f[72] = prefactor_units[120] * fwd_A[120]
               * exp(fwd_beta[120] * tc[0] - activation_units[120] * fwd_Ea[120] * invT);
    k_f[73] = prefactor_units[122] * fwd_A[122]
               * exp(fwd_beta[122] * tc[0] - activation_units[122] * fwd_Ea[122] * invT);
    k_f[74] = prefactor_units[123] * fwd_A[123]
               * exp(fwd_beta[123] * tc[0] - activation_units[123] * fwd_Ea[123] * invT);
    k_f[75] = prefactor_units[124] * fwd_A[124]
               * exp(fwd_beta[124] * tc[0] - activation_units[124] * fwd_Ea[124] * invT);
    k_f[76] = prefactor_units[125] * fwd_A[125]
               * exp(fwd_beta[125] * tc[0] - activation_units[125] * fwd_Ea[125] * invT);
    k_f[77] = prefactor_units[126] * fwd_A[126]
               * exp(fwd_beta[126] * tc[0] - activation_units[126] * fwd_Ea[126] * invT);
    k_f[78] = prefactor_units[127] * fwd_A[127]
               * exp(fwd_beta[127] * tc[0] - activation_units[127] * fwd_Ea[127] * invT);
    k_f[79] = prefactor_units[128] * fwd_A[128]
               * exp(fwd_beta[128] * tc[0] - activation_units[128] * fwd_Ea[128] * invT);
    k_f[80] = prefactor_units[129] * fwd_A[129]
               * exp(fwd_beta[129] * tc[0] - activation_units[129] * fwd_Ea[129] * invT);
    k_f[81] = prefactor_units[130] * fwd_A[130]
               * exp(fwd_beta[130] * tc[0] - activation_units[130] * fwd_Ea[130] * invT);
    k_f[82] = prefactor_units[131] * fwd_A[131]
               * exp(fwd_beta[131] * tc[0] - activation_units[131] * fwd_Ea[131] * invT);
    k_f[83] = prefactor_units[132] * fwd_A[132]
               * exp(fwd_beta[132] * tc[0] - activation_units[132] * fwd_Ea[132] * invT);
    k_f[84] = prefactor_units[133] * fwd_A[133]
               * exp(fwd_beta[133] * tc[0] - activation_units[133] * fwd_Ea[133] * invT);
    k_f[85] = prefactor_units[134] * fwd_A[134]
               * exp(fwd_beta[134] * tc[0] - activation_units[134] * fwd_Ea[134] * invT);
    k_f[86] = prefactor_units[135] * fwd_A[135]
               * exp(fwd_beta[135] * tc[0] - activation_units[135] * fwd_Ea[135] * invT);
    k_f[87] = prefactor_units[136] * fwd_A[136]
               * exp(fwd_beta[136] * tc[0] - activation_units[136] * fwd_Ea[136] * invT);
    k_f[88] = prefactor_units[137] * fwd_A[137]
               * exp(fwd_beta[137] * tc[0] - activation_units[137] * fwd_Ea[137] * invT);
    k_f[89] = prefactor_units[138] * fwd_A[138]
               * exp(fwd_beta[138] * tc[0] - activation_units[138] * fwd_Ea[138] * invT);
    k_f[90] = prefactor_units[146] * fwd_A[146]
               * exp(fwd_beta[146] * tc[0] - activation_units[146] * fwd_Ea[146] * invT);
    k_f[91] = prefactor_units[150] * fwd_A[150]
               * exp(fwd_beta[150] * tc[0] - activation_units[150] * fwd_Ea[150] * invT);
    k_f[92] = prefactor_units[154] * fwd_A[154]
               * exp(fwd_beta[154] * tc[0] - activation_units[154] * fwd_Ea[154] * invT);
    k_f[93] = prefactor_units[155] * fwd_A[155]
               * exp(fwd_beta[155] * tc[0] - activation_units[155] * fwd_Ea[155] * invT);
    k_f[94] = prefactor_units[156] * fwd_A[156]
               * exp(fwd_beta[156] * tc[0] - activation_units[156] * fwd_Ea[156] * invT);
    k_f[95] = prefactor_units[157] * fwd_A[157]
               * exp(fwd_beta[157] * tc[0] - activation_units[157] * fwd_Ea[157] * invT);
    k_f[96] = prefactor_units[158] * fwd_A[158]
               * exp(fwd_beta[158] * tc[0] - activation_units[158] * fwd_Ea[158] * invT);
    k_f[97] = prefactor_units[159] * fwd_A[159]
               * exp(fwd_beta[159] * tc[0] - activation_units[159] * fwd_Ea[159] * invT);
    k_f[98] = prefactor_units[160] * fwd_A[160]
               * exp(fwd_beta[160] * tc[0] - activation_units[160] * fwd_Ea[160] * invT);
    k_f[99] = prefactor_units[161] * fwd_A[161]
               * exp(fwd_beta[161] * tc[0] - activation_units[161] * fwd_Ea[161] * invT);
    k_f[100] = prefactor_units[162] * fwd_A[162]
               * exp(fwd_beta[162] * tc[0] - activation_units[162] * fwd_Ea[162] * invT);
    k_f[101] = prefactor_units[164] * fwd_A[164]
               * exp(fwd_beta[164] * tc[0] - activation_units[164] * fwd_Ea[164] * invT);
    k_f[102] = prefactor_units[166] * fwd_A[166]
               * exp(fwd_beta[166] * tc[0] - activation_units[166] * fwd_Ea[166] * invT);
    k_f[103] = prefactor_units[170] * fwd_A[170]
               * exp(fwd_beta[170] * tc[0] - activation_units[170] * fwd_Ea[170] * invT);
    k_f[104] = prefactor_units[176] * fwd_A[176]
               * exp(fwd_beta[176] * tc[0] - activation_units[176] * fwd_Ea[176] * invT);
    k_f[105] = prefactor_units[179] * fwd_A[179]
               * exp(fwd_beta[179] * tc[0] - activation_units[179] * fwd_Ea[179] * invT);
    k_f[106] = prefactor_units[181] * fwd_A[181]
               * exp(fwd_beta[181] * tc[0] - activation_units[181] * fwd_Ea[181] * invT);
    k_f[107] = prefactor_units[186] * fwd_A[186]
               * exp(fwd_beta[186] * tc[0] - activation_units[186] * fwd_Ea[186] * invT);
    k_f[108] = prefactor_units[187] * fwd_A[187]
               * exp(fwd_beta[187] * tc[0] - activation_units[187] * fwd_Ea[187] * invT);
    k_f[109] = prefactor_units[188] * fwd_A[188]
               * exp(fwd_beta[188] * tc[0] - activation_units[188] * fwd_Ea[188] * invT);
    k_f[110] = prefactor_units[189] * fwd_A[189]
               * exp(fwd_beta[189] * tc[0] - activation_units[189] * fwd_Ea[189] * invT);
    k_f[111] = prefactor_units[190] * fwd_A[190]
               * exp(fwd_beta[190] * tc[0] - activation_units[190] * fwd_Ea[190] * invT);
    k_f[112] = prefactor_units[191] * fwd_A[191]
               * exp(fwd_beta[191] * tc[0] - activation_units[191] * fwd_Ea[191] * invT);
    k_f[113] = prefactor_units[192] * fwd_A[192]
               * exp(fwd_beta[192] * tc[0] - activation_units[192] * fwd_Ea[192] * invT);
    k_f[114] = prefactor_units[193] * fwd_A[193]
               * exp(fwd_beta[193] * tc[0] - activation_units[193] * fwd_Ea[193] * invT);
    k_f[115] = prefactor_units[194] * fwd_A[194]
               * exp(fwd_beta[194] * tc[0] - activation_units[194] * fwd_Ea[194] * invT);
    k_f[116] = prefactor_units[195] * fwd_A[195]
               * exp(fwd_beta[195] * tc[0] - activation_units[195] * fwd_Ea[195] * invT);
    k_f[117] = prefactor_units[196] * fwd_A[196]
               * exp(fwd_beta[196] * tc[0] - activation_units[196] * fwd_Ea[196] * invT);
    k_f[118] = prefactor_units[197] * fwd_A[197]
               * exp(fwd_beta[197] * tc[0] - activation_units[197] * fwd_Ea[197] * invT);
    k_f[119] = prefactor_units[198] * fwd_A[198]
               * exp(fwd_beta[198] * tc[0] - activation_units[198] * fwd_Ea[198] * invT);
    k_f[120] = prefactor_units[199] * fwd_A[199]
               * exp(fwd_beta[199] * tc[0] - activation_units[199] * fwd_Ea[199] * invT);
    k_f[121] = prefactor_units[200] * fwd_A[200]
               * exp(fwd_beta[200] * tc[0] - activation_units[200] * fwd_Ea[200] * invT);
    k_f[122] = prefactor_units[201] * fwd_A[201]
               * exp(fwd_beta[201] * tc[0] - activation_units[201] * fwd_Ea[201] * invT);
    k_f[123] = prefactor_units[202] * fwd_A[202]
               * exp(fwd_beta[202] * tc[0] - activation_units[202] * fwd_Ea[202] * invT);
    k_f[124] = prefactor_units[203] * fwd_A[203]
               * exp(fwd_beta[203] * tc[0] - activation_units[203] * fwd_Ea[203] * invT);
    k_f[125] = prefactor_units[204] * fwd_A[204]
               * exp(fwd_beta[204] * tc[0] - activation_units[204] * fwd_Ea[204] * invT);
    k_f[126] = prefactor_units[205] * fwd_A[205]
               * exp(fwd_beta[205] * tc[0] - activation_units[205] * fwd_Ea[205] * invT);
    k_f[127] = prefactor_units[206] * fwd_A[206]
               * exp(fwd_beta[206] * tc[0] - activation_units[206] * fwd_Ea[206] * invT);
    k_f[128] = prefactor_units[207] * fwd_A[207]
               * exp(fwd_beta[207] * tc[0] - activation_units[207] * fwd_Ea[207] * invT);
    k_f[129] = prefactor_units[209] * fwd_A[209]
               * exp(fwd_beta[209] * tc[0] - activation_units[209] * fwd_Ea[209] * invT);
    k_f[130] = prefactor_units[211] * fwd_A[211]
               * exp(fwd_beta[211] * tc[0] - activation_units[211] * fwd_Ea[211] * invT);
    k_f[131] = prefactor_units[213] * fwd_A[213]
               * exp(fwd_beta[213] * tc[0] - activation_units[213] * fwd_Ea[213] * invT);
    k_f[132] = prefactor_units[214] * fwd_A[214]
               * exp(fwd_beta[214] * tc[0] - activation_units[214] * fwd_Ea[214] * invT);

    return;
}

void comp_Kc_qss(double *  tc, double invT, double *  Kc)
{
    /*compute the Gibbs free energy */
    double g_RT[32], g_RT_qss[20];
    gibbs(g_RT, tc);
    gibbs_qss(g_RT_qss, tc);

    /*Reaction 6 */
    Kc[0] = g_RT[10] - g_RT[17] + g_RT_qss[2];
    /*Reaction 8 */
    Kc[1] = -g_RT[3] - g_RT[11] + g_RT_qss[3];
    /*Reaction 9 */
    Kc[2] = -g_RT[3] - g_RT[20] + g_RT_qss[7];
    /*Reaction 10 */
    Kc[3] = g_RT[3] + g_RT[22] - g_RT_qss[4];
    /*Reaction 13 */
    Kc[4] = g_RT[12] - g_RT_qss[2];
    /*Reaction 14 */
    Kc[5] = -g_RT[12] + g_RT_qss[2];
    /*Reaction 15 */
    Kc[6] = -g_RT[3] - g_RT[10] + g_RT_qss[1];
    /*Reaction 18 */
    Kc[7] = -g_RT[11] - g_RT[13] + g_RT_qss[9];
    /*Reaction 34 */
    Kc[8] = -g_RT[1] + g_RT[6] + g_RT_qss[0] - g_RT_qss[1];
    /*Reaction 35 */
    Kc[9] = -2.000000*g_RT[3] + g_RT[6] - g_RT[9] + g_RT_qss[2];
    /*Reaction 36 */
    Kc[10] = -g_RT[5] + g_RT[6] - g_RT[10] + g_RT_qss[2];
    /*Reaction 37 */
    Kc[11] = g_RT[1] - 2.000000*g_RT[3] - g_RT[10] + g_RT_qss[2];
    /*Reaction 38 */
    Kc[12] = -g_RT[1] + g_RT[6] - g_RT[11] + g_RT_qss[2];
    /*Reaction 39 */
    Kc[13] = -g_RT[2] + g_RT[3] - g_RT_qss[0] + g_RT_qss[2];
    /*Reaction 40 */
    Kc[14] = g_RT[2] - g_RT[3] + g_RT_qss[0] - g_RT_qss[2];
    /*Reaction 41 */
    Kc[15] = g_RT[4] - g_RT[5] - g_RT_qss[0] + g_RT_qss[2];
    /*Reaction 42 */
    Kc[16] = -g_RT[4] + g_RT[5] + g_RT_qss[0] - g_RT_qss[2];
    /*Reaction 43 */
    Kc[17] = -g_RT[2] + g_RT[6] - g_RT[9] + g_RT_qss[2];
    /*Reaction 44 */
    Kc[18] = -g_RT[2] + g_RT[3] + g_RT[12] - g_RT_qss[0];
    /*Reaction 53 */
    Kc[19] = -g_RT[4] + g_RT[7] + g_RT[13] - g_RT_qss[3];
    /*Reaction 56 */
    Kc[20] = -g_RT[2] + g_RT[3] + g_RT[13] - g_RT_qss[2];
    /*Reaction 57 */
    Kc[21] = g_RT[2] - g_RT[3] - g_RT[13] + g_RT_qss[2];
    /*Reaction 58 */
    Kc[22] = -g_RT[3] + 2.000000*g_RT[13] - g_RT_qss[4];
    /*Reaction 59 */
    Kc[23] = g_RT[4] - g_RT[5] + g_RT[13] - g_RT_qss[2];
    /*Reaction 60 */
    Kc[24] = -g_RT[4] + g_RT[5] - g_RT[13] + g_RT_qss[2];
    /*Reaction 68 */
    Kc[25] = g_RT[6] - g_RT[7] - g_RT[10] + g_RT_qss[1];
    /*Reaction 69 */
    Kc[26] = g_RT[1] - g_RT[3] - g_RT[9] + g_RT_qss[1];
    /*Reaction 70 */
    Kc[27] = g_RT[4] - g_RT[5] - g_RT[10] + g_RT_qss[1];
    /*Reaction 71 */
    Kc[28] = -g_RT[2] + g_RT[3] - g_RT[10] + g_RT_qss[1];
    /*Reaction 72 */
    Kc[29] = g_RT[1] - g_RT[4] - g_RT[10] + g_RT_qss[1];
    /*Reaction 73 */
    Kc[30] = -g_RT[10] + g_RT[13] - g_RT[14] + g_RT_qss[1];
    /*Reaction 74 */
    Kc[31] = g_RT[4] - g_RT[5] + g_RT[11] - g_RT_qss[1];
    /*Reaction 75 */
    Kc[32] = g_RT[1] - g_RT[4] + g_RT[11] - g_RT_qss[1];
    /*Reaction 76 */
    Kc[33] = -g_RT[2] + g_RT[3] + g_RT[11] - g_RT_qss[1];
    /*Reaction 77 */
    Kc[34] = g_RT[11] + g_RT[13] - g_RT[14] - g_RT_qss[1];
    /*Reaction 78 */
    Kc[35] = -g_RT[11] - g_RT[15] + 2.000000*g_RT_qss[3];
    /*Reaction 79 */
    Kc[36] = g_RT[6] - g_RT[7] - g_RT[11] + g_RT_qss[3];
    /*Reaction 80 */
    Kc[37] = g_RT[2] - g_RT[3] - g_RT[15] + g_RT_qss[3];
    /*Reaction 81 */
    Kc[38] = g_RT[4] - g_RT[5] + g_RT[15] - g_RT_qss[3];
    /*Reaction 85 */
    Kc[39] = -g_RT[4] + g_RT[18] - g_RT_qss[1];
    /*Reaction 86 */
    Kc[40] = g_RT[4] - g_RT[18] + g_RT_qss[1];
    /*Reaction 89 */
    Kc[41] = -g_RT[6] + 2.000000*g_RT[19] - 2.000000*g_RT_qss[3];
    /*Reaction 90 */
    Kc[42] = g_RT[13] + g_RT[19] - 2.000000*g_RT_qss[3];
    /*Reaction 92 */
    Kc[43] = -g_RT[6] + g_RT[7] + g_RT[19] - g_RT_qss[5];
    /*Reaction 93 */
    Kc[44] = -g_RT[4] - g_RT_qss[3] + g_RT_qss[5];
    /*Reaction 94 */
    Kc[45] = g_RT[1] - g_RT[10] + g_RT[20] - g_RT_qss[2];
    /*Reaction 95 */
    Kc[46] = g_RT[1] - g_RT[3] + g_RT[20] - g_RT_qss[6];
    /*Reaction 96 */
    Kc[47] = -g_RT[2] + g_RT[3] - g_RT[20] + g_RT_qss[7];
    /*Reaction 97 */
    Kc[48] = -g_RT[1] + g_RT[6] + g_RT_qss[7] - g_RT_qss[8];
    /*Reaction 98 */
    Kc[49] = g_RT[13] - g_RT[21] + g_RT_qss[7];
    /*Reaction 99 */
    Kc[50] = g_RT[6] - g_RT[7] - g_RT[20] + g_RT_qss[7];
    /*Reaction 100 */
    Kc[51] = g_RT[6] - g_RT[11] - g_RT_qss[1] + g_RT_qss[7];
    /*Reaction 101 */
    Kc[52] = g_RT[13] - g_RT[14] + g_RT[22] - g_RT_qss[7];
    /*Reaction 102 */
    Kc[53] = g_RT[1] - g_RT[13] + g_RT[22] - g_RT_qss[1];
    /*Reaction 103 */
    Kc[54] = g_RT[4] - g_RT[5] + g_RT[22] - g_RT_qss[7];
    /*Reaction 104 */
    Kc[55] = g_RT[1] - g_RT[3] + g_RT[22] - g_RT_qss[8];
    /*Reaction 105 */
    Kc[56] = -g_RT[2] + g_RT[3] + g_RT[22] - g_RT_qss[7];
    /*Reaction 106 */
    Kc[57] = g_RT[2] - g_RT[3] - g_RT[22] + g_RT_qss[7];
    /*Reaction 107 */
    Kc[58] = g_RT[3] - g_RT[16] + g_RT_qss[4];
    /*Reaction 108 */
    Kc[59] = g_RT[19] - g_RT_qss[3] + g_RT_qss[4] - g_RT_qss[9];
    /*Reaction 109 */
    Kc[60] = -g_RT[4] + g_RT[7] + g_RT_qss[4] - g_RT_qss[9];
    /*Reaction 110 */
    Kc[61] = g_RT[6] - g_RT[7] - g_RT[22] + g_RT_qss[4];
    /*Reaction 111 */
    Kc[62] = g_RT[1] - g_RT[4] + g_RT[16] - g_RT_qss[4];
    /*Reaction 112 */
    Kc[63] = g_RT[4] - g_RT[5] + g_RT[16] - g_RT_qss[4];
    /*Reaction 113 */
    Kc[64] = -g_RT[2] + g_RT[3] + g_RT[16] - g_RT_qss[4];
    /*Reaction 114 */
    Kc[65] = g_RT[1] - g_RT[3] - 2.000000*g_RT[10] + g_RT_qss[6];
    /*Reaction 115 */
    Kc[66] = g_RT[4] - 2.000000*g_RT_qss[1] + g_RT_qss[6];
    /*Reaction 116 */
    Kc[67] = g_RT[6] - g_RT[9] - g_RT_qss[1] + g_RT_qss[6];
    /*Reaction 117 */
    Kc[68] = g_RT[3] - g_RT[10] - g_RT[12] + g_RT_qss[6];
    /*Reaction 118 */
    Kc[69] = -g_RT[3] + g_RT[10] + g_RT[12] - g_RT_qss[6];
    /*Reaction 119 */
    Kc[70] = g_RT[1] - g_RT[4] + g_RT[17] - g_RT_qss[6];
    /*Reaction 120 */
    Kc[71] = -g_RT[2] + g_RT[3] + g_RT[17] - g_RT_qss[6];
    /*Reaction 121 */
    Kc[72] = g_RT[2] - g_RT[3] - g_RT[17] + g_RT_qss[6];
    /*Reaction 123 */
    Kc[73] = g_RT[1] - g_RT[9] + g_RT[17] - g_RT_qss[2];
    /*Reaction 124 */
    Kc[74] = g_RT[4] - g_RT[5] + g_RT[17] - g_RT_qss[6];
    /*Reaction 125 */
    Kc[75] = -g_RT[4] + g_RT[6] - g_RT[10] - g_RT[11] + g_RT_qss[8];
    /*Reaction 126 */
    Kc[76] = -g_RT[3] - g_RT[17] + g_RT_qss[8];
    /*Reaction 127 */
    Kc[77] = g_RT[3] + g_RT[17] - g_RT_qss[8];
    /*Reaction 128 */
    Kc[78] = -g_RT[6] - g_RT_qss[4] + g_RT_qss[10];
    /*Reaction 129 */
    Kc[79] = g_RT[6] + g_RT_qss[4] - g_RT_qss[10];
    /*Reaction 130 */
    Kc[80] = -g_RT[7] - g_RT[22] + g_RT_qss[10];
    /*Reaction 131 */
    Kc[81] = -g_RT[3] + g_RT[6] - g_RT[10] - g_RT_qss[6] + g_RT_qss[11];
    /*Reaction 132 */
    Kc[82] = g_RT[4] - g_RT[20] - g_RT_qss[1] + g_RT_qss[11];
    /*Reaction 133 */
    Kc[83] = g_RT[6] - g_RT[17] - g_RT_qss[1] + g_RT_qss[12];
    /*Reaction 134 */
    Kc[84] = -g_RT[6] + g_RT[7] - g_RT[24] + g_RT_qss[12];
    /*Reaction 135 */
    Kc[85] = -g_RT[2] + g_RT[3] - g_RT_qss[11] + g_RT_qss[12];
    /*Reaction 136 */
    Kc[86] = g_RT[4] - g_RT[5] - g_RT_qss[11] + g_RT_qss[12];
    /*Reaction 137 */
    Kc[87] = -g_RT[4] + g_RT[5] + g_RT_qss[11] - g_RT_qss[12];
    /*Reaction 138 */
    Kc[88] = -g_RT[2] + g_RT[3] + g_RT[24] - g_RT_qss[12];
    /*Reaction 139 */
    Kc[89] = g_RT[4] - g_RT[5] + g_RT[24] - g_RT_qss[12];
    /*Reaction 147 */
    Kc[90] = g_RT[11] - g_RT[21] + g_RT[25] - g_RT_qss[1];
    /*Reaction 151 */
    Kc[91] = g_RT[1] + g_RT[21] - g_RT_qss[1] - g_RT_qss[4];
    /*Reaction 155 */
    Kc[92] = g_RT[6] - g_RT[7] - g_RT[21] + g_RT_qss[13];
    /*Reaction 156 */
    Kc[93] = -g_RT[13] - g_RT[22] + g_RT_qss[13];
    /*Reaction 157 */
    Kc[94] = g_RT[13] + g_RT[22] - g_RT_qss[13];
    /*Reaction 158 */
    Kc[95] = -g_RT[3] - g_RT[21] + g_RT_qss[13];
    /*Reaction 159 */
    Kc[96] = g_RT[3] + g_RT[21] - g_RT_qss[13];
    /*Reaction 160 */
    Kc[97] = -g_RT[6] + g_RT[26] - g_RT_qss[13];
    /*Reaction 161 */
    Kc[98] = g_RT[6] - g_RT[26] + g_RT_qss[13];
    /*Reaction 162 */
    Kc[99] = g_RT[27] - 2.000000*g_RT_qss[7];
    /*Reaction 163 */
    Kc[100] = -g_RT[27] + 2.000000*g_RT_qss[7];
    /*Reaction 165 */
    Kc[101] = g_RT[4] - g_RT[17] + g_RT[27] - g_RT_qss[4];
    /*Reaction 167 */
    Kc[102] = g_RT[3] - g_RT[22] + g_RT[27] - g_RT_qss[7];
    /*Reaction 171 */
    Kc[103] = -g_RT[16] - g_RT[27] + g_RT[28] + g_RT_qss[4];
    /*Reaction 177 */
    Kc[104] = -g_RT[22] + g_RT[28] - g_RT_qss[7];
    /*Reaction 180 */
    Kc[105] = g_RT[4] - g_RT[11] + g_RT[29] - g_RT_qss[13];
    /*Reaction 182 */
    Kc[106] = g_RT[1] - g_RT[23] + g_RT[29] - g_RT_qss[4];
    /*Reaction 187 */
    Kc[107] = -g_RT[3] - g_RT[29] + g_RT_qss[14];
    /*Reaction 188 */
    Kc[108] = g_RT[3] + g_RT[29] - g_RT_qss[14];
    /*Reaction 189 */
    Kc[109] = -g_RT[22] - g_RT_qss[4] + g_RT_qss[14];
    /*Reaction 190 */
    Kc[110] = -g_RT[6] - g_RT_qss[14] + g_RT_qss[15];
    /*Reaction 191 */
    Kc[111] = g_RT[6] + g_RT_qss[14] - g_RT_qss[15];
    /*Reaction 192 */
    Kc[112] = -g_RT[13] - g_RT[27] + g_RT_qss[16];
    /*Reaction 193 */
    Kc[113] = -g_RT[22] - g_RT[25] + g_RT_qss[16];
    /*Reaction 194 */
    Kc[114] = g_RT[4] - g_RT[5] - g_RT_qss[16] + g_RT_qss[17];
    /*Reaction 195 */
    Kc[115] = -g_RT[2] + g_RT[3] - g_RT_qss[16] + g_RT_qss[17];
    /*Reaction 196 */
    Kc[116] = -g_RT[25] - g_RT_qss[4] + g_RT_qss[17];
    /*Reaction 197 */
    Kc[117] = g_RT[25] + g_RT_qss[4] - g_RT_qss[17];
    /*Reaction 198 */
    Kc[118] = g_RT[1] - g_RT[4] - g_RT_qss[16] + g_RT_qss[17];
    /*Reaction 199 */
    Kc[119] = -g_RT[21] - g_RT_qss[4] + g_RT_qss[18];
    /*Reaction 200 */
    Kc[120] = -g_RT[22] - g_RT_qss[13] + g_RT_qss[18];
    /*Reaction 201 */
    Kc[121] = -g_RT[3] - g_RT_qss[17] + g_RT_qss[18];
    /*Reaction 202 */
    Kc[122] = -g_RT[25] - g_RT_qss[13] + g_RT_qss[19];
    /*Reaction 203 */
    Kc[123] = g_RT[4] - g_RT[11] - g_RT_qss[18] + g_RT_qss[19];
    /*Reaction 204 */
    Kc[124] = -g_RT[13] + g_RT[30] - g_RT_qss[19];
    /*Reaction 205 */
    Kc[125] = -g_RT[21] + g_RT[30] - g_RT_qss[14];
    /*Reaction 206 */
    Kc[126] = -g_RT[29] + g_RT[30] - g_RT_qss[13];
    /*Reaction 207 */
    Kc[127] = -g_RT[22] + g_RT[30] - g_RT_qss[18];
    /*Reaction 208 */
    Kc[128] = g_RT[30] - g_RT_qss[4] - g_RT_qss[17];
    /*Reaction 210 */
    Kc[129] = g_RT[19] - g_RT[30] + g_RT[31] - g_RT_qss[5];
    /*Reaction 212 */
    Kc[130] = g_RT[31] - g_RT_qss[13] - g_RT_qss[14];
    /*Reaction 214 */
    Kc[131] = g_RT[31] - g_RT_qss[4] - g_RT_qss[18];
    /*Reaction 215 */
    Kc[132] = -g_RT[15] - g_RT[30] + g_RT[31] + g_RT_qss[3];

#ifdef __INTEL_COMPILER
     #pragma simd
#endif
    for (int i=0; i<133; ++i) {
        Kc[i] = exp(Kc[i]);
    };

    /*reference concentration: P_atm / (RT) in inverse mol/m^3 */
    double refC = 101325 / 8.31446 * invT;
    double refCinv = 1 / refC;

    Kc[0] *= refCinv;
    Kc[1] *= refC;
    Kc[2] *= refC;
    Kc[3] *= refCinv;
    Kc[6] *= refC;
    Kc[7] *= refC;
    Kc[9] *= refC;
    Kc[11] *= refC;
    Kc[39] *= refC;
    Kc[40] *= refCinv;
    Kc[41] *= refC;
    Kc[44] *= refC;
    Kc[49] *= refCinv;
    Kc[58] *= refCinv;
    Kc[65] *= refC;
    Kc[75] *= refC;
    Kc[76] *= refC;
    Kc[77] *= refCinv;
    Kc[78] *= refC;
    Kc[79] *= refCinv;
    Kc[80] *= refC;
    Kc[81] *= refC;
    Kc[93] *= refC;
    Kc[94] *= refCinv;
    Kc[95] *= refC;
    Kc[96] *= refCinv;
    Kc[97] *= refC;
    Kc[98] *= refCinv;
    Kc[99] *= refC;
    Kc[100] *= refCinv;
    Kc[104] *= refC;
    Kc[107] *= refC;
    Kc[108] *= refCinv;
    Kc[109] *= refC;
    Kc[110] *= refC;
    Kc[111] *= refCinv;
    Kc[112] *= refC;
    Kc[113] *= refC;
    Kc[116] *= refC;
    Kc[117] *= refCinv;
    Kc[119] *= refC;
    Kc[120] *= refC;
    Kc[121] *= refC;
    Kc[122] *= refC;
    Kc[124] *= refC;
    Kc[125] *= refC;
    Kc[126] *= refC;
    Kc[127] *= refC;
    Kc[128] *= refC;
    Kc[130] *= refC;
    Kc[131] *= refC;

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
    qf_co[3] = sc[3]*sc[22];
    qr_co[3] = 1.0;

    /*reaction 13: CH2GSG + M => CH2 + M */
    qf_co[4] = sc[12];
    qr_co[4] = 0.0;

    /*reaction 14: CH2 + M => CH2GSG + M */
    qf_co[5] = 1.0;
    qr_co[5] = 0.0;

    /*reaction 15: HCO + M => H + CO + M */
    qf_co[6] = 1.0;
    qr_co[6] = 0.0;

    /*reaction 18: C2H5O + M => CH3 + CH2O + M */
    qf_co[7] = 1.0;
    qr_co[7] = 0.0;

    /*reaction 34: CH + O2 => HCO + O */
    qf_co[8] = sc[6];
    qr_co[8] = 0.0;

    /*reaction 35: CH2 + O2 => CO2 + 2.000000 H */
    qf_co[9] = sc[6];
    qr_co[9] = 0.0;

    /*reaction 36: CH2 + O2 => CO + H2O */
    qf_co[10] = sc[6];
    qr_co[10] = 0.0;

    /*reaction 37: CH2 + O => CO + 2.000000 H */
    qf_co[11] = sc[1];
    qr_co[11] = 0.0;

    /*reaction 38: CH2 + O2 => CH2O + O */
    qf_co[12] = sc[6];
    qr_co[12] = 0.0;

    /*reaction 39: CH2 + H => CH + H2 */
    qf_co[13] = sc[3];
    qr_co[13] = 0.0;

    /*reaction 40: CH + H2 => CH2 + H */
    qf_co[14] = sc[2];
    qr_co[14] = 0.0;

    /*reaction 41: CH2 + OH => CH + H2O */
    qf_co[15] = sc[4];
    qr_co[15] = 0.0;

    /*reaction 42: CH + H2O => CH2 + OH */
    qf_co[16] = sc[5];
    qr_co[16] = 0.0;

    /*reaction 43: CH2 + O2 => CO2 + H2 */
    qf_co[17] = sc[6];
    qr_co[17] = 0.0;

    /*reaction 44: CH2GSG + H => CH + H2 */
    qf_co[18] = sc[3]*sc[12];
    qr_co[18] = 0.0;

    /*reaction 53: CH3 + HO2 => CH3O + OH */
    qf_co[19] = sc[7]*sc[13];
    qr_co[19] = 0.0;

    /*reaction 56: CH3 + H => CH2 + H2 */
    qf_co[20] = sc[3]*sc[13];
    qr_co[20] = 0.0;

    /*reaction 57: CH2 + H2 => CH3 + H */
    qf_co[21] = sc[2];
    qr_co[21] = 0.0;

    /*reaction 58: 2.000000 CH3 <=> H + C2H5 */
    qf_co[22] = pow(sc[13], 2.000000);
    qr_co[22] = sc[3];

    /*reaction 59: CH3 + OH => CH2 + H2O */
    qf_co[23] = sc[4]*sc[13];
    qr_co[23] = 0.0;

    /*reaction 60: CH2 + H2O => CH3 + OH */
    qf_co[24] = sc[5];
    qr_co[24] = 0.0;

    /*reaction 68: HCO + O2 => CO + HO2 */
    qf_co[25] = sc[6];
    qr_co[25] = 0.0;

    /*reaction 69: HCO + O => CO2 + H */
    qf_co[26] = sc[1];
    qr_co[26] = 0.0;

    /*reaction 70: HCO + OH => CO + H2O */
    qf_co[27] = sc[4];
    qr_co[27] = 0.0;

    /*reaction 71: HCO + H => CO + H2 */
    qf_co[28] = sc[3];
    qr_co[28] = 0.0;

    /*reaction 72: HCO + O => CO + OH */
    qf_co[29] = sc[1];
    qr_co[29] = 0.0;

    /*reaction 73: HCO + CH3 => CH4 + CO */
    qf_co[30] = 1.0*sc[13];
    qr_co[30] = 0.0;

    /*reaction 74: CH2O + OH => HCO + H2O */
    qf_co[31] = sc[4]*sc[11];
    qr_co[31] = 0.0;

    /*reaction 75: CH2O + O => HCO + OH */
    qf_co[32] = sc[1]*sc[11];
    qr_co[32] = 0.0;

    /*reaction 76: CH2O + H => HCO + H2 */
    qf_co[33] = sc[3]*sc[11];
    qr_co[33] = 0.0;

    /*reaction 77: CH2O + CH3 => HCO + CH4 */
    qf_co[34] = sc[11]*sc[13];
    qr_co[34] = 0.0;

    /*reaction 78: 2.000000 CH3O => CH3OH + CH2O */
    qf_co[35] = 1.0;
    qr_co[35] = 0.0;

    /*reaction 79: CH3O + O2 => CH2O + HO2 */
    qf_co[36] = sc[6];
    qr_co[36] = 0.0;

    /*reaction 80: CH3O + H2 => CH3OH + H */
    qf_co[37] = sc[2];
    qr_co[37] = 0.0;

    /*reaction 81: CH3OH + OH => CH3O + H2O */
    qf_co[38] = sc[4]*sc[15];
    qr_co[38] = 0.0;

    /*reaction 85: HOCHO => HCO + OH */
    qf_co[39] = sc[18];
    qr_co[39] = 0.0;

    /*reaction 86: HCO + OH => HOCHO */
    qf_co[40] = sc[4];
    qr_co[40] = 0.0;

    /*reaction 89: 2.000000 CH3O2 => O2 + 2.000000 CH3O */
    qf_co[41] = pow(sc[19], 2.000000);
    qr_co[41] = 0.0;

    /*reaction 90: CH3O2 + CH3 => 2.000000 CH3O */
    qf_co[42] = sc[13]*sc[19];
    qr_co[42] = 0.0;

    /*reaction 92: CH3O2 + HO2 => CH3O2H + O2 */
    qf_co[43] = sc[7]*sc[19];
    qr_co[43] = 0.0;

    /*reaction 93: CH3O2H => CH3O + OH */
    qf_co[44] = 1.0;
    qr_co[44] = 0.0;

    /*reaction 94: C2H2 + O => CH2 + CO */
    qf_co[45] = sc[1]*sc[20];
    qr_co[45] = 0.0;

    /*reaction 95: C2H2 + O => HCCO + H */
    qf_co[46] = sc[1]*sc[20];
    qr_co[46] = 0.0;

    /*reaction 96: C2H3 + H => C2H2 + H2 */
    qf_co[47] = sc[3];
    qr_co[47] = 0.0;

    /*reaction 97: C2H3 + O2 => CH2CHO + O */
    qf_co[48] = sc[6];
    qr_co[48] = 0.0;

    /*reaction 98: C2H3 + CH3 => C3H6 */
    qf_co[49] = sc[13];
    qr_co[49] = 0.0;

    /*reaction 99: C2H3 + O2 => C2H2 + HO2 */
    qf_co[50] = sc[6];
    qr_co[50] = 0.0;

    /*reaction 100: C2H3 + O2 => CH2O + HCO */
    qf_co[51] = sc[6];
    qr_co[51] = 0.0;

    /*reaction 101: C2H4 + CH3 => C2H3 + CH4 */
    qf_co[52] = sc[13]*sc[22];
    qr_co[52] = 0.0;

    /*reaction 102: C2H4 + O => CH3 + HCO */
    qf_co[53] = sc[1]*sc[22];
    qr_co[53] = 0.0;

    /*reaction 103: C2H4 + OH => C2H3 + H2O */
    qf_co[54] = sc[4]*sc[22];
    qr_co[54] = 0.0;

    /*reaction 104: C2H4 + O => CH2CHO + H */
    qf_co[55] = sc[1]*sc[22];
    qr_co[55] = 0.0;

    /*reaction 105: C2H4 + H => C2H3 + H2 */
    qf_co[56] = sc[3]*sc[22];
    qr_co[56] = 0.0;

    /*reaction 106: C2H3 + H2 => C2H4 + H */
    qf_co[57] = sc[2];
    qr_co[57] = 0.0;

    /*reaction 107: H + C2H5 => C2H6 */
    qf_co[58] = sc[3];
    qr_co[58] = 0.0;

    /*reaction 108: CH3O2 + C2H5 => CH3O + C2H5O */
    qf_co[59] = 1.0*sc[19];
    qr_co[59] = 0.0;

    /*reaction 109: C2H5 + HO2 => C2H5O + OH */
    qf_co[60] = sc[7];
    qr_co[60] = 0.0;

    /*reaction 110: C2H5 + O2 => C2H4 + HO2 */
    qf_co[61] = sc[6];
    qr_co[61] = 0.0;

    /*reaction 111: C2H6 + O => C2H5 + OH */
    qf_co[62] = sc[1]*sc[16];
    qr_co[62] = 0.0;

    /*reaction 112: C2H6 + OH => C2H5 + H2O */
    qf_co[63] = sc[4]*sc[16];
    qr_co[63] = 0.0;

    /*reaction 113: C2H6 + H => C2H5 + H2 */
    qf_co[64] = sc[3]*sc[16];
    qr_co[64] = 0.0;

    /*reaction 114: HCCO + O => H + 2.000000 CO */
    qf_co[65] = sc[1];
    qr_co[65] = 0.0;

    /*reaction 115: HCCO + OH => 2.000000 HCO */
    qf_co[66] = sc[4];
    qr_co[66] = 0.0;

    /*reaction 116: HCCO + O2 => CO2 + HCO */
    qf_co[67] = sc[6];
    qr_co[67] = 0.0;

    /*reaction 117: HCCO + H => CH2GSG + CO */
    qf_co[68] = sc[3];
    qr_co[68] = 0.0;

    /*reaction 118: CH2GSG + CO => HCCO + H */
    qf_co[69] = sc[10]*sc[12];
    qr_co[69] = 0.0;

    /*reaction 119: CH2CO + O => HCCO + OH */
    qf_co[70] = sc[1]*sc[17];
    qr_co[70] = 0.0;

    /*reaction 120: CH2CO + H => HCCO + H2 */
    qf_co[71] = sc[3]*sc[17];
    qr_co[71] = 0.0;

    /*reaction 121: HCCO + H2 => CH2CO + H */
    qf_co[72] = sc[2];
    qr_co[72] = 0.0;

    /*reaction 123: CH2CO + O => CH2 + CO2 */
    qf_co[73] = sc[1]*sc[17];
    qr_co[73] = 0.0;

    /*reaction 124: CH2CO + OH => HCCO + H2O */
    qf_co[74] = sc[4]*sc[17];
    qr_co[74] = 0.0;

    /*reaction 125: CH2CHO + O2 => CH2O + CO + OH */
    qf_co[75] = sc[6];
    qr_co[75] = 0.0;

    /*reaction 126: CH2CHO => CH2CO + H */
    qf_co[76] = 1.0;
    qr_co[76] = 0.0;

    /*reaction 127: CH2CO + H => CH2CHO */
    qf_co[77] = sc[3]*sc[17];
    qr_co[77] = 0.0;

    /*reaction 128: C2H5O2 => C2H5 + O2 */
    qf_co[78] = 1.0;
    qr_co[78] = 0.0;

    /*reaction 129: C2H5 + O2 => C2H5O2 */
    qf_co[79] = sc[6];
    qr_co[79] = 0.0;

    /*reaction 130: C2H5O2 => C2H4 + HO2 */
    qf_co[80] = 1.0;
    qr_co[80] = 0.0;

    /*reaction 131: C3H2 + O2 => HCCO + CO + H */
    qf_co[81] = sc[6];
    qr_co[81] = 0.0;

    /*reaction 132: C3H2 + OH => C2H2 + HCO */
    qf_co[82] = sc[4];
    qr_co[82] = 0.0;

    /*reaction 133: C3H3 + O2 => CH2CO + HCO */
    qf_co[83] = sc[6];
    qr_co[83] = 0.0;

    /*reaction 134: C3H3 + HO2 => C3H4XA + O2 */
    qf_co[84] = sc[7];
    qr_co[84] = 0.0;

    /*reaction 135: C3H3 + H => C3H2 + H2 */
    qf_co[85] = sc[3];
    qr_co[85] = 0.0;

    /*reaction 136: C3H3 + OH => C3H2 + H2O */
    qf_co[86] = sc[4];
    qr_co[86] = 0.0;

    /*reaction 137: C3H2 + H2O => C3H3 + OH */
    qf_co[87] = sc[5];
    qr_co[87] = 0.0;

    /*reaction 138: C3H4XA + H => C3H3 + H2 */
    qf_co[88] = sc[3]*sc[24];
    qr_co[88] = 0.0;

    /*reaction 139: C3H4XA + OH => C3H3 + H2O */
    qf_co[89] = sc[4]*sc[24];
    qr_co[89] = 0.0;

    /*reaction 147: C3H5XA + CH2O => C3H6 + HCO */
    qf_co[90] = sc[11]*sc[25];
    qr_co[90] = 0.0;

    /*reaction 151: C3H6 + O => C2H5 + HCO */
    qf_co[91] = sc[1]*sc[21];
    qr_co[91] = 0.0;

    /*reaction 155: NXC3H7 + O2 => C3H6 + HO2 */
    qf_co[92] = sc[6];
    qr_co[92] = 0.0;

    /*reaction 156: NXC3H7 => CH3 + C2H4 */
    qf_co[93] = 1.0;
    qr_co[93] = 0.0;

    /*reaction 157: CH3 + C2H4 => NXC3H7 */
    qf_co[94] = sc[13]*sc[22];
    qr_co[94] = 0.0;

    /*reaction 158: NXC3H7 => H + C3H6 */
    qf_co[95] = 1.0;
    qr_co[95] = 0.0;

    /*reaction 159: H + C3H6 => NXC3H7 */
    qf_co[96] = sc[3]*sc[21];
    qr_co[96] = 0.0;

    /*reaction 160: NXC3H7O2 => NXC3H7 + O2 */
    qf_co[97] = sc[26];
    qr_co[97] = 0.0;

    /*reaction 161: NXC3H7 + O2 => NXC3H7O2 */
    qf_co[98] = sc[6];
    qr_co[98] = 0.0;

    /*reaction 162: C4H6 => 2.000000 C2H3 */
    qf_co[99] = sc[27];
    qr_co[99] = 0.0;

    /*reaction 163: 2.000000 C2H3 => C4H6 */
    qf_co[100] = 1.0;
    qr_co[100] = 0.0;

    /*reaction 165: C4H6 + OH => C2H5 + CH2CO */
    qf_co[101] = sc[4]*sc[27];
    qr_co[101] = 0.0;

    /*reaction 167: C4H6 + H => C2H3 + C2H4 */
    qf_co[102] = sc[3]*sc[27];
    qr_co[102] = 0.0;

    /*reaction 171: C2H5 + C4H7 => C4H6 + C2H6 */
    qf_co[103] = 1.0*sc[28];
    qr_co[103] = 0.0;

    /*reaction 177: C4H7 => C2H4 + C2H3 */
    qf_co[104] = sc[28];
    qr_co[104] = 0.0;

    /*reaction 180: C4H8X1 + OH => NXC3H7 + CH2O */
    qf_co[105] = sc[4]*sc[29];
    qr_co[105] = 0.0;

    /*reaction 182: C4H8X1 + O => CH3CO + C2H5 */
    qf_co[106] = sc[1]*sc[29];
    qr_co[106] = 0.0;

    /*reaction 187: PXC4H9 => C4H8X1 + H */
    qf_co[107] = 1.0;
    qr_co[107] = 0.0;

    /*reaction 188: C4H8X1 + H => PXC4H9 */
    qf_co[108] = sc[3]*sc[29];
    qr_co[108] = 0.0;

    /*reaction 189: PXC4H9 => C2H5 + C2H4 */
    qf_co[109] = 1.0;
    qr_co[109] = 0.0;

    /*reaction 190: PXC4H9O2 => PXC4H9 + O2 */
    qf_co[110] = 1.0;
    qr_co[110] = 0.0;

    /*reaction 191: PXC4H9 + O2 => PXC4H9O2 */
    qf_co[111] = sc[6];
    qr_co[111] = 0.0;

    /*reaction 192: C5H9 => C4H6 + CH3 */
    qf_co[112] = 1.0;
    qr_co[112] = 0.0;

    /*reaction 193: C5H9 => C3H5XA + C2H4 */
    qf_co[113] = 1.0;
    qr_co[113] = 0.0;

    /*reaction 194: C5H10X1 + OH => C5H9 + H2O */
    qf_co[114] = sc[4];
    qr_co[114] = 0.0;

    /*reaction 195: C5H10X1 + H => C5H9 + H2 */
    qf_co[115] = sc[3];
    qr_co[115] = 0.0;

    /*reaction 196: C5H10X1 => C2H5 + C3H5XA */
    qf_co[116] = 1.0;
    qr_co[116] = 0.0;

    /*reaction 197: C2H5 + C3H5XA => C5H10X1 */
    qf_co[117] = 1.0*sc[25];
    qr_co[117] = 0.0;

    /*reaction 198: C5H10X1 + O => C5H9 + OH */
    qf_co[118] = sc[1];
    qr_co[118] = 0.0;

    /*reaction 199: C5H11X1 => C3H6 + C2H5 */
    qf_co[119] = 1.0;
    qr_co[119] = 0.0;

    /*reaction 200: C5H11X1 => C2H4 + NXC3H7 */
    qf_co[120] = 1.0;
    qr_co[120] = 0.0;

    /*reaction 201: C5H11X1 <=> C5H10X1 + H */
    qf_co[121] = 1.0;
    qr_co[121] = sc[3];

    /*reaction 202: C6H12X1 => NXC3H7 + C3H5XA */
    qf_co[122] = 1.0;
    qr_co[122] = 0.0;

    /*reaction 203: C6H12X1 + OH => C5H11X1 + CH2O */
    qf_co[123] = sc[4];
    qr_co[123] = 0.0;

    /*reaction 204: C7H15X2 => C6H12X1 + CH3 */
    qf_co[124] = sc[30];
    qr_co[124] = 0.0;

    /*reaction 205: C7H15X2 => PXC4H9 + C3H6 */
    qf_co[125] = sc[30];
    qr_co[125] = 0.0;

    /*reaction 206: C7H15X2 => C4H8X1 + NXC3H7 */
    qf_co[126] = sc[30];
    qr_co[126] = 0.0;

    /*reaction 207: C7H15X2 => C5H11X1 + C2H4 */
    qf_co[127] = sc[30];
    qr_co[127] = 0.0;

    /*reaction 208: C7H15X2 => C2H5 + C5H10X1 */
    qf_co[128] = sc[30];
    qr_co[128] = 0.0;

    /*reaction 210: NXC7H16 + CH3O2 => C7H15X2 + CH3O2H */
    qf_co[129] = sc[19]*sc[31];
    qr_co[129] = 0.0;

    /*reaction 212: NXC7H16 => PXC4H9 + NXC3H7 */
    qf_co[130] = sc[31];
    qr_co[130] = 0.0;

    /*reaction 214: NXC7H16 => C5H11X1 + C2H5 */
    qf_co[131] = sc[31];
    qr_co[131] = 0.0;

    /*reaction 215: NXC7H16 + CH3O => C7H15X2 + CH3OH */
    qf_co[132] = 1.0*sc[31];
    qr_co[132] = 0.0;

    double T = tc[1];

    /*compute the mixture concentration */
    double mixture = 0.0;
    for (int i = 0; i < 32; ++i) {
        mixture += sc[i];
    }

    double Corr[133];
    for (int i = 0; i < 133; ++i) {
        Corr[i] = 1.0;
    }

    /* troe */
    {
        double alpha[4];
        alpha[0] = mixture + (TB[5][4] - 1)*sc[12] + (TB[5][5] - 1)*sc[28] + (TB[5][8] - 1)*sc[23] + (TB[5][10] - 1)*sc[30];
        alpha[1] = mixture + (TB[7][4] - 1)*sc[12] + (TB[7][5] - 1)*sc[28] + (TB[7][8] - 1)*sc[23] + (TB[7][10] - 1)*sc[30];
        alpha[2] = mixture + (TB[8][2] - 1)*sc[10] + (TB[8][3] - 1)*sc[2] + (TB[8][5] - 1)*sc[12] + (TB[8][6] - 1)*sc[28] + (TB[8][9] - 1)*sc[5] + (TB[8][10] - 1)*sc[9] + (TB[8][13] - 1)*sc[23] + (TB[8][14] - 1)*sc[30];
        alpha[3] = mixture + (TB[9][4] - 1)*sc[12] + (TB[9][5] - 1)*sc[28] + (TB[9][8] - 1)*sc[23] + (TB[9][10] - 1)*sc[30];
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
    }

    /* simple three-body correction */
    {
        double alpha;
        alpha = mixture + (TB[12][4] - 1)*sc[12] + (TB[12][5] - 1)*sc[28] + (TB[12][8] - 1)*sc[23] + (TB[12][10] - 1)*sc[30];
        Corr[4] = alpha;
        alpha = mixture + (TB[13][4] - 1)*sc[12] + (TB[13][5] - 1)*sc[28] + (TB[13][8] - 1)*sc[23] + (TB[13][10] - 1)*sc[30];
        Corr[5] = alpha;
        alpha = mixture + (TB[14][2] - 1)*sc[10] + (TB[14][3] - 1)*sc[2] + (TB[14][5] - 1)*sc[12] + (TB[14][6] - 1)*sc[28] + (TB[14][9] - 1)*sc[5] + (TB[14][10] - 1)*sc[9] + (TB[14][13] - 1)*sc[23] + (TB[14][14] - 1)*sc[30];
        Corr[6] = alpha;
        alpha = mixture + (TB[17][4] - 1)*sc[12] + (TB[17][5] - 1)*sc[28] + (TB[17][8] - 1)*sc[23] + (TB[17][10] - 1)*sc[30];
        Corr[7] = alpha;
    }

    for (int i=0; i<133; i++)
    {
        qf_co[i] *= Corr[i] * k_f_save_qss[i];
        qr_co[i] *= Corr[i] * k_f_save_qss[i] / Kc_save_qss[i];
    }

    return;
}

void comp_qss_sc(double * sc, double * sc_qss, double * tc, double invT)
{

    double  qf_co[133], qr_co[133];
    double epsilon = 1e-16;

    comp_qss_coeff(qf_co, qr_co, sc, tc, invT);

    /*QSS species 5: CH3O2H */

    double CH3O2H_num = epsilon - qf_co[43] - qf_co[129];
    double CH3O2H_denom = epsilon - qf_co[44];

    sc_qss[5] = CH3O2H_num/CH3O2H_denom;

    /*QSS species 7: C2H3 */

    double C2H3_num = epsilon - qf_co[52] - qf_co[54] - qf_co[56] - qf_co[99] - qf_co[102] - qf_co[104];
    double C2H3_denom = epsilon - qf_co[2] - qf_co[47] - qf_co[48] - qf_co[49] - qf_co[50] - qf_co[51] - qf_co[57] - qf_co[100];

    sc_qss[7] = C2H3_num/C2H3_denom;

    /*QSS species 19: C6H12X1 */

    double C6H12X1_num = epsilon - qf_co[124];
    double C6H12X1_denom = epsilon - qf_co[122] - qf_co[123];

    sc_qss[19] = C6H12X1_num/C6H12X1_denom;

    /*QSS species 0: CH */

    double CH_num = epsilon - qf_co[18];
    double CH_denom = epsilon - qf_co[8] - qf_co[14] - qf_co[16];
    double CH_rhs = CH_num/CH_denom;

    double CH_CH2 = (epsilon + qf_co[13] + qf_co[15])/CH_denom;

    /*QSS species 2: CH2 */

    double CH2_num = epsilon - qf_co[4] - qf_co[20] - qf_co[23] - qf_co[45] - qf_co[73];
    double CH2_denom = epsilon - qf_co[0] - qf_co[5] - qf_co[9] - qf_co[10] - qf_co[11] - qf_co[12] - qf_co[13] - qf_co[15] - qf_co[17] - qf_co[21] - qf_co[24];
    double CH2_rhs = CH2_num/CH2_denom;

    double CH2_CH = (epsilon + qf_co[14] + qf_co[16])/CH2_denom;

    sc_qss[2] = (CH2_rhs - CH_rhs * CH2_CH) / (1 - CH_CH2 * CH2_CH);

    sc_qss[0] = CH_rhs - (CH_CH2 * sc_qss[2]);

    /*QSS species 11: C3H2 */

    double C3H2_num = epsilon ;
    double C3H2_denom = epsilon - qf_co[81] - qf_co[82] - qf_co[87];
    double C3H2_rhs = C3H2_num/C3H2_denom;

    double C3H2_C3H3 = (epsilon + qf_co[85] + qf_co[86])/C3H2_denom;

    /*QSS species 12: C3H3 */

    double C3H3_num = epsilon - qf_co[88] - qf_co[89];
    double C3H3_denom = epsilon - qf_co[83] - qf_co[84] - qf_co[85] - qf_co[86];
    double C3H3_rhs = C3H3_num/C3H3_denom;

    double C3H3_C3H2 = (epsilon + qf_co[87])/C3H3_denom;

    sc_qss[12] = (C3H3_rhs - C3H2_rhs * C3H3_C3H2) / (1 - C3H2_C3H3 * C3H3_C3H2);

    sc_qss[11] = C3H2_rhs - (C3H2_C3H3 * sc_qss[12]);

    /*QSS species 14: PXC4H9 */

    double PXC4H9_num = epsilon - qf_co[108] - qf_co[125] - qf_co[130]*sc_qss[13];
    double PXC4H9_denom = epsilon - qf_co[107] - qf_co[109] - qf_co[111];
    double PXC4H9_rhs = PXC4H9_num/PXC4H9_denom;

    double PXC4H9_PXC4H9O2 = (epsilon + qf_co[110])/PXC4H9_denom;

    /*QSS species 15: PXC4H9O2 */

    double PXC4H9O2_num = epsilon ;
    double PXC4H9O2_denom = epsilon - qf_co[110];
    double PXC4H9O2_rhs = PXC4H9O2_num/PXC4H9O2_denom;

    double PXC4H9O2_PXC4H9 = (epsilon + qf_co[111])/PXC4H9O2_denom;

    sc_qss[15] = (PXC4H9O2_rhs - PXC4H9_rhs * PXC4H9O2_PXC4H9) / (1 - PXC4H9_PXC4H9O2 * PXC4H9O2_PXC4H9);

    sc_qss[14] = PXC4H9_rhs - (PXC4H9_PXC4H9O2 * sc_qss[15]);

    /*QSS species 6: HCCO */

    double HCCO_num = epsilon - qf_co[46] - qf_co[69] - qf_co[70] - qf_co[71] - qf_co[74] - qf_co[81]*sc_qss[11];
    double HCCO_denom = epsilon - qf_co[65] - qf_co[66] - qf_co[67] - qf_co[68] - qf_co[72];

    sc_qss[6] = HCCO_num/HCCO_denom;

    /*QSS species 8: CH2CHO */

    double CH2CHO_num = epsilon - qf_co[48]*sc_qss[7] - qf_co[55] - qf_co[77];
    double CH2CHO_denom = epsilon - qf_co[75] - qf_co[76];

    sc_qss[8] = CH2CHO_num/CH2CHO_denom;

    /*QSS species 4: C2H5 */

    double C2H5_num = epsilon - qf_co[3] - qf_co[22] - qf_co[62] - qf_co[63] - qf_co[64] - qf_co[91]*sc_qss[1] - qf_co[101] - qf_co[106] - qf_co[109]*sc_qss[14];
    double C2H5_denom = epsilon - qr_co[3] - qr_co[22] - qf_co[58] - qf_co[59] - qf_co[60] - qf_co[61] - qf_co[79] - qf_co[103] - qf_co[117];
    double C2H5_rhs = C2H5_num/C2H5_denom;

    double C2H5_C2H5O2 = (epsilon + qf_co[78] + qf_co[116] + qf_co[119] + qf_co[128] + qf_co[131])/C2H5_denom;
    double C2H5_C5H10X1 = (epsilon + qf_co[78] + qf_co[116] + qf_co[119] + qf_co[128] + qf_co[131])/C2H5_denom;
    double C2H5_C5H11X1 = (epsilon + qf_co[78] + qf_co[116] + qf_co[119] + qf_co[128] + qf_co[131])/C2H5_denom;

    /*QSS species 10: C2H5O2 */

    double C2H5O2_num = epsilon ;
    double C2H5O2_denom = epsilon - qf_co[78] - qf_co[80];
    double C2H5O2_rhs = C2H5O2_num/C2H5O2_denom;

    double C2H5O2_C2H5 = (epsilon + qf_co[79])/C2H5O2_denom;
    double C2H5O2_C5H10X1 = (epsilon + qf_co[79])/C2H5O2_denom;
    double C2H5O2_C5H11X1 = (epsilon + qf_co[79])/C2H5O2_denom;

    /*QSS species 17: C5H10X1 */

    double C5H10X1_num = epsilon ;
    double C5H10X1_denom = epsilon - qf_co[114] - qf_co[115] - qf_co[116] - qf_co[118] - qr_co[121];
    double C5H10X1_rhs = C5H10X1_num/C5H10X1_denom;

    double C5H10X1_C2H5 = (epsilon + qf_co[117] + qf_co[121] + qf_co[128])/C5H10X1_denom;
    double C5H10X1_C2H5O2 = (epsilon + qf_co[117] + qf_co[121] + qf_co[128])/C5H10X1_denom;
    double C5H10X1_C5H11X1 = (epsilon + qf_co[117] + qf_co[121] + qf_co[128])/C5H10X1_denom;

    /*QSS species 18: C5H11X1 */

    double C5H11X1_num = epsilon - qf_co[123]*sc_qss[19] - qf_co[127];
    double C5H11X1_denom = epsilon - qf_co[119] - qf_co[120] - qf_co[121];
    double C5H11X1_rhs = C5H11X1_num/C5H11X1_denom;

    double C5H11X1_C2H5 = (epsilon + qr_co[121] + qf_co[131])/C5H11X1_denom;
    double C5H11X1_C2H5O2 = (epsilon + qr_co[121] + qf_co[131])/C5H11X1_denom;
    double C5H11X1_C5H10X1 = (epsilon + qr_co[121] + qf_co[131])/C5H11X1_denom;

    sc_qss[18] = (((C5H11X1_rhs - C2H5_rhs * C5H11X1_C2H5) - (C2H5O2_rhs - C2H5_rhs * C2H5O2_C2H5) * (C5H11X1_C2H5O2 - C2H5_C2H5O2 * C5H11X1_C2H5) / (1 - C2H5_C2H5O2 * C2H5O2_C2H5)) - ((C5H10X1_rhs - C2H5_rhs * C5H10X1_C2H5) - (C2H5O2_rhs - C2H5_rhs * C2H5O2_C2H5) * (C5H10X1_C2H5O2 - C2H5_C2H5O2 * C5H10X1_C2H5) / (1 - C2H5_C2H5O2 * C2H5O2_C2H5)) * ((C5H11X1_C5H10X1 - C2H5_C5H10X1 * C5H11X1_C2H5) - (C2H5O2_C5H10X1 - C2H5_C5H10X1 * C2H5O2_C2H5) * (C5H11X1_C2H5O2 - C2H5_C2H5O2 * C5H11X1_C2H5)/ (1 - C2H5_C2H5O2 * C2H5O2_C2H5)) / ((1 - C2H5_C5H10X1 * C5H10X1_C2H5) - (C2H5O2_C5H10X1 - C2H5_C5H10X1 * C2H5O2_C2H5) * (C5H10X1_C2H5O2 - C2H5_C2H5O2 * C5H10X1_C2H5)/ (1 - C2H5_C2H5O2 * C2H5O2_C2H5))) / (((1 - C2H5_C5H11X1 * C5H11X1_C2H5) - (C2H5O2_C5H11X1 - C2H5_C5H11X1 * C2H5O2_C2H5) * (C5H11X1_C2H5O2 - C2H5_C2H5O2 * C5H11X1_C2H5)/ (1 - C2H5_C2H5O2 * C2H5O2_C2H5)) - ((C5H10X1_C5H11X1 - C2H5_C5H11X1 * C5H10X1_C2H5) - (C2H5O2_C5H11X1 - C2H5_C5H11X1 * C2H5O2_C2H5) * (C5H10X1_C2H5O2 - C2H5_C2H5O2 * C5H10X1_C2H5)/ (1 - C2H5_C2H5O2 * C2H5O2_C2H5)) * ((C5H11X1_C5H10X1 - C2H5_C5H10X1 * C5H11X1_C2H5) - (C2H5O2_C5H10X1 - C2H5_C5H10X1 * C2H5O2_C2H5) * (C5H11X1_C2H5O2 - C2H5_C2H5O2 * C5H11X1_C2H5)/ (1 - C2H5_C2H5O2 * C2H5O2_C2H5))/ ((1 - C2H5_C5H10X1 * C5H10X1_C2H5) - (C2H5O2_C5H10X1 - C2H5_C5H10X1 * C2H5O2_C2H5) * (C5H10X1_C2H5O2 - C2H5_C2H5O2 * C5H10X1_C2H5)/ (1 - C2H5_C2H5O2 * C2H5O2_C2H5)));

    sc_qss[17] = (((C5H10X1_rhs - C2H5_rhs * C5H10X1_C2H5) - (C2H5O2_rhs - C2H5_rhs * C2H5O2_C2H5) * (C5H10X1_C2H5O2 - C2H5_C2H5O2 * C5H10X1_C2H5) / (1 - C2H5_C2H5O2 * C2H5O2_C2H5)) - (((C5H10X1_C5H11X1 - C2H5_C5H11X1 * C5H10X1_C2H5) - (C2H5O2_C5H11X1 - C2H5_C5H11X1 * C2H5O2_C2H5) * (C5H10X1_C2H5O2 - C2H5_C2H5O2 * C5H10X1_C2H5)/ (1 - C2H5_C2H5O2 * C2H5O2_C2H5)) * sc_qss[18])) / ((1 - C2H5_C5H10X1 * C5H10X1_C2H5) - (C2H5O2_C5H10X1 - C2H5_C5H10X1 * C2H5O2_C2H5) * (C5H10X1_C2H5O2 - C2H5_C2H5O2 * C5H10X1_C2H5)/ (1 - C2H5_C2H5O2 * C2H5O2_C2H5));

    sc_qss[10] = ((C2H5O2_rhs - C2H5_rhs * C2H5O2_C2H5) - ((C2H5O2_C5H11X1 - C2H5_C5H11X1 * C2H5O2_C2H5) * sc_qss[18] + (C2H5O2_C5H10X1 - C2H5_C5H10X1 * C2H5O2_C2H5) * sc_qss[17])) / (1 - C2H5_C2H5O2 * C2H5O2_C2H5);

    sc_qss[4] = C2H5_rhs - (C2H5_C5H11X1 * sc_qss[18] + C2H5_C5H10X1 * sc_qss[17] + C2H5_C2H5O2 * sc_qss[10]);

    /*QSS species 1: HCO */

    double HCO_num = epsilon - qf_co[8]*sc_qss[0] - qf_co[31] - qf_co[32] - qf_co[33] - qf_co[34] - qf_co[39] - qf_co[51]*sc_qss[7] - qf_co[53] - qf_co[66]*sc_qss[6] - qf_co[67]*sc_qss[6] - qf_co[82]*sc_qss[11] - qf_co[83]*sc_qss[12] - qf_co[90] - qf_co[91]*sc_qss[4];
    double HCO_denom = epsilon - qf_co[6] - qf_co[25] - qf_co[26] - qf_co[27] - qf_co[28] - qf_co[29] - qf_co[30] - qf_co[40];

    sc_qss[1] = HCO_num/HCO_denom;

    /*QSS species 3: CH3O */

    double CH3O_num = epsilon - qf_co[19] - qf_co[38] - qf_co[41] - qf_co[42] - qf_co[44]*sc_qss[5] - qf_co[59]*sc_qss[4];
    double CH3O_denom = epsilon - qf_co[1] - qf_co[35] - qf_co[36] - qf_co[37] - qf_co[132];

    sc_qss[3] = CH3O_num/CH3O_denom;

    /*QSS species 9: C2H5O */

    double C2H5O_num = epsilon - qf_co[59]*sc_qss[4] - qf_co[60]*sc_qss[4];
    double C2H5O_denom = epsilon - qf_co[7];

    sc_qss[9] = C2H5O_num/C2H5O_denom;

    /*QSS species 13: NXC3H7 */

    double NXC3H7_num = epsilon - qf_co[94] - qf_co[96] - qf_co[97] - qf_co[105] - qf_co[120]*sc_qss[18] - qf_co[122]*sc_qss[19] - qf_co[126] - qf_co[130]*sc_qss[14];
    double NXC3H7_denom = epsilon - qf_co[92] - qf_co[93] - qf_co[95] - qf_co[98];

    sc_qss[13] = NXC3H7_num/NXC3H7_denom;

    /*QSS species 16: C5H9 */

    double C5H9_num = epsilon - qf_co[114]*sc_qss[17] - qf_co[115]*sc_qss[17] - qf_co[118]*sc_qss[17];
    double C5H9_denom = epsilon - qf_co[112] - qf_co[113];

    sc_qss[16] = C5H9_num/C5H9_denom;


    return;
}

/*compute an approx to the reaction Jacobian (for preconditioning) */
AMREX_GPU_HOST_DEVICE void DWDOT_SIMPLIFIED(double *  J, double *  sc, double *  Tp, int * HP)
{
    double c[32];

    for (int k=0; k<32; k++) {
        c[k] = 1.e6 * sc[k];
    }

    aJacobian_precond(J, c, *Tp, *HP);

    /* dwdot[k]/dT */
    /* dTdot/d[X] */
    for (int k=0; k<32; k++) {
        J[1056+k] *= 1.e-6;
        J[k*33+32] *= 1.e6;
    }

    return;
}

/*compute the reaction Jacobian */
AMREX_GPU_HOST_DEVICE void DWDOT(double *  J, double *  sc, double *  Tp, int * consP)
{
    double c[32];

    for (int k=0; k<32; k++) {
        c[k] = 1.e6 * sc[k];
    }

    aJacobian(J, c, *Tp, *consP);

    /* dwdot[k]/dT */
    /* dTdot/d[X] */
    for (int k=0; k<32; k++) {
        J[1056+k] *= 1.e-6;
        J[k*33+32] *= 1.e6;
    }

    return;
}

/*compute the sparsity pattern of the chemistry Jacobian */
AMREX_GPU_HOST_DEVICE void SPARSITY_INFO( int * nJdata, int * consP, int NCELLS)
{
    double c[32];
    double J[1089];

    for (int k=0; k<32; k++) {
        c[k] = 1.0/ 32.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    int nJdata_tmp = 0;
    for (int k=0; k<33; k++) {
        for (int l=0; l<33; l++) {
            if(J[ 33 * k + l] != 0.0){
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
    double c[32];
    double J[1089];

    for (int k=0; k<32; k++) {
        c[k] = 1.0/ 32.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    int nJdata_tmp = 0;
    for (int k=0; k<33; k++) {
        for (int l=0; l<33; l++) {
            if(k == l){
                nJdata_tmp = nJdata_tmp + 1;
            } else {
                if(J[ 33 * k + l] != 0.0){
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
    double c[32];
    double J[1089];

    for (int k=0; k<32; k++) {
        c[k] = 1.0/ 32.000000 ;
    }

    aJacobian_precond(J, c, 1500.0, *consP);

    int nJdata_tmp = 0;
    for (int k=0; k<33; k++) {
        for (int l=0; l<33; l++) {
            if(k == l){
                nJdata_tmp = nJdata_tmp + 1;
            } else {
                if(J[ 33 * k + l] != 0.0){
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
    double c[32];
    double J[1089];
    int offset_row;
    int offset_col;

    for (int k=0; k<32; k++) {
        c[k] = 1.0/ 32.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    colPtrs[0] = 0;
    int nJdata_tmp = 0;
    for (int nc=0; nc<NCELLS; nc++) {
        offset_row = nc * 33;
        offset_col = nc * 33;
        for (int k=0; k<33; k++) {
            for (int l=0; l<33; l++) {
                if(J[33*k + l] != 0.0) {
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
    double c[32];
    double J[1089];
    int offset;

    for (int k=0; k<32; k++) {
        c[k] = 1.0/ 32.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    if (base == 1) {
        rowPtrs[0] = 1;
        int nJdata_tmp = 1;
        for (int nc=0; nc<NCELLS; nc++) {
            offset = nc * 33;
            for (int l=0; l<33; l++) {
                for (int k=0; k<33; k++) {
                    if(J[33*k + l] != 0.0) {
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
            offset = nc * 33;
            for (int l=0; l<33; l++) {
                for (int k=0; k<33; k++) {
                    if(J[33*k + l] != 0.0) {
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
    double c[32];
    double J[1089];
    int offset;

    for (int k=0; k<32; k++) {
        c[k] = 1.0/ 32.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    if (base == 1) {
        rowPtr[0] = 1;
        int nJdata_tmp = 1;
        for (int nc=0; nc<NCELLS; nc++) {
            offset = nc * 33;
            for (int l=0; l<33; l++) {
                for (int k=0; k<33; k++) {
                    if (k == l) {
                        colVals[nJdata_tmp-1] = l+1 + offset; 
                        nJdata_tmp = nJdata_tmp + 1; 
                    } else {
                        if(J[33*k + l] != 0.0) {
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
            offset = nc * 33;
            for (int l=0; l<33; l++) {
                for (int k=0; k<33; k++) {
                    if (k == l) {
                        colVals[nJdata_tmp] = l + offset; 
                        nJdata_tmp = nJdata_tmp + 1; 
                    } else {
                        if(J[33*k + l] != 0.0) {
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
    double c[32];
    double J[1089];

    for (int k=0; k<32; k++) {
        c[k] = 1.0/ 32.000000 ;
    }

    aJacobian_precond(J, c, 1500.0, *consP);

    colPtrs[0] = 0;
    int nJdata_tmp = 0;
    for (int k=0; k<33; k++) {
        for (int l=0; l<33; l++) {
            if (k == l) {
                rowVals[nJdata_tmp] = l; 
                indx[nJdata_tmp] = 33*k + l;
                nJdata_tmp = nJdata_tmp + 1; 
            } else {
                if(J[33*k + l] != 0.0) {
                    rowVals[nJdata_tmp] = l; 
                    indx[nJdata_tmp] = 33*k + l;
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
    double c[32];
    double J[1089];

    for (int k=0; k<32; k++) {
        c[k] = 1.0/ 32.000000 ;
    }

    aJacobian_precond(J, c, 1500.0, *consP);

    if (base == 1) {
        rowPtr[0] = 1;
        int nJdata_tmp = 1;
        for (int l=0; l<33; l++) {
            for (int k=0; k<33; k++) {
                if (k == l) {
                    colVals[nJdata_tmp-1] = l+1; 
                    nJdata_tmp = nJdata_tmp + 1; 
                } else {
                    if(J[33*k + l] != 0.0) {
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
        for (int l=0; l<33; l++) {
            for (int k=0; k<33; k++) {
                if (k == l) {
                    colVals[nJdata_tmp] = l; 
                    nJdata_tmp = nJdata_tmp + 1; 
                } else {
                    if(J[33*k + l] != 0.0) {
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
    for (int i=0; i<1089; i++) {
        J[i] = 0.0;
    }
}
#endif


/*compute an approx to the reaction Jacobian */
AMREX_GPU_HOST_DEVICE void aJacobian_precond(double *  J, double *  sc, double T, int HP)
{
    for (int i=0; i<1089; i++) {
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
        /*species 12: CH2GSG */
        species[12] =
            -1.69908900e-04
            +2.05073800e-06 * tc[1]
            +7.47765300e-09 * tc[2]
            -7.92506400e-12 * tc[3];
        /*species 13: CH3 */
        species[13] =
            +1.11241000e-02
            -3.36044000e-05 * tc[1]
            +4.86548700e-08 * tc[2]
            -2.34598120e-11 * tc[3];
        /*species 14: CH4 */
        species[14] =
            +1.74766800e-02
            -5.56681800e-05 * tc[1]
            +9.14912400e-08 * tc[2]
            -4.89572400e-11 * tc[3];
        /*species 15: CH3OH */
        species[15] =
            +7.34150800e-03
            +1.43401020e-05 * tc[1]
            -2.63795820e-08 * tc[2]
            +9.56228000e-12 * tc[3];
        /*species 17: CH2CO */
        species[17] =
            +1.21187100e-02
            -4.69009200e-06 * tc[1]
            -1.94000550e-08 * tc[2]
            +1.56225960e-11 * tc[3];
        /*species 20: C2H2 */
        species[20] =
            +1.51904500e-02
            -3.23263800e-05 * tc[1]
            +2.72369760e-08 * tc[2]
            -7.65098400e-12 * tc[3];
        /*species 22: C2H4 */
        species[22] =
            +2.79616300e-02
            -6.77735400e-05 * tc[1]
            +8.35545600e-08 * tc[2]
            -3.89515160e-11 * tc[3];
        /*species 23: CH3CO */
        species[23] =
            +9.77822000e-03
            +9.04289600e-06 * tc[1]
            -2.70283860e-08 * tc[2]
            +1.27748720e-11 * tc[3];
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
        /*species 12: CH2GSG */
        species[12] =
            +2.06678800e-03
            -3.82823200e-07 * tc[1]
            -3.31401900e-10 * tc[2]
            +8.08540000e-14 * tc[3];
        /*species 13: CH3 */
        species[13] =
            +6.13797400e-03
            -4.46069000e-06 * tc[1]
            +1.13554830e-09 * tc[2]
            -9.80863600e-14 * tc[3];
        /*species 14: CH4 */
        species[14] =
            +1.02372400e-02
            -7.75025800e-06 * tc[1]
            +2.03567550e-09 * tc[2]
            -1.80136920e-13 * tc[3];
        /*species 15: CH3OH */
        species[15] =
            +9.37659300e-03
            -6.10050800e-06 * tc[1]
            +1.30763790e-09 * tc[2]
            -8.89889200e-14 * tc[3];
        /*species 17: CH2CO */
        species[17] =
            +5.80484000e-03
            -3.84190800e-06 * tc[1]
            +8.38345500e-10 * tc[2]
            -5.83547200e-14 * tc[3];
        /*species 20: C2H2 */
        species[20] =
            +5.37603900e-03
            -3.82563400e-06 * tc[1]
            +9.85913700e-10 * tc[2]
            -8.62684000e-14 * tc[3];
        /*species 22: C2H4 */
        species[22] =
            +1.14851800e-02
            -8.83677000e-06 * tc[1]
            +2.35338030e-09 * tc[2]
            -2.10673920e-13 * tc[3];
        /*species 23: CH3CO */
        species[23] =
            +8.44988600e-03
            -5.70829400e-06 * tc[1]
            +1.27151280e-09 * tc[2]
            -9.07361600e-14 * tc[3];
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
        /*species 16: C2H6 */
        species[16] =
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
        /*species 16: C2H6 */
        species[16] =
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
        /*species 18: HOCHO */
        species[18] =
            +1.63363016e-02
            -2.12514842e-05 * tc[1]
            +9.96398931e-09 * tc[2]
            -1.60870441e-12 * tc[3];
    } else {
        /*species 18: HOCHO */
        species[18] =
            +5.14289368e-03
            -3.64477026e-06 * tc[1]
            +8.69157489e-10 * tc[2]
            -6.83568796e-14 * tc[3];
    }

    /*species with midpoint at T=1385 kelvin */
    if (T < 1385) {
        /*species 19: CH3O2 */
        species[19] =
            +1.00873599e-02
            -6.43012368e-06 * tc[1]
            +6.28227801e-10 * tc[2]
            +1.67335641e-13 * tc[3];
    } else {
        /*species 19: CH3O2 */
        species[19] =
            +7.90728626e-03
            -5.36492468e-06 * tc[1]
            +1.24167401e-09 * tc[2]
            -9.56029320e-14 * tc[3];
    }

    /*species with midpoint at T=1388 kelvin */
    if (T < 1388) {
        /*species 21: C3H6 */
        species[21] =
            +2.89107662e-02
            -3.09773616e-05 * tc[1]
            +1.16644263e-08 * tc[2]
            -1.35156141e-12 * tc[3];
    } else {
        /*species 21: C3H6 */
        species[21] =
            +1.37023634e-02
            -9.32499466e-06 * tc[1]
            +2.16376321e-09 * tc[2]
            -1.66948050e-13 * tc[3];
    }

    /*species with midpoint at T=1400 kelvin */
    if (T < 1400) {
        /*species 24: C3H4XA */
        species[24] =
            +1.63343700e-02
            -3.52990000e-06 * tc[1]
            -1.39420950e-08 * tc[2]
            +6.91652400e-12 * tc[3];
    } else {
        /*species 24: C3H4XA */
        species[24] =
            +5.30213800e-03
            -7.40223600e-07 * tc[1]
            -9.07915800e-10 * tc[2]
            +2.03583240e-13 * tc[3];
    }

    /*species with midpoint at T=1397 kelvin */
    if (T < 1397) {
        /*species 25: C3H5XA */
        species[25] =
            +3.34559100e-02
            -5.06802054e-05 * tc[1]
            +3.08597262e-08 * tc[2]
            -6.93033360e-12 * tc[3];
    } else {
        /*species 25: C3H5XA */
        species[25] =
            +1.12695483e-02
            -7.67585728e-06 * tc[1]
            +1.78217736e-09 * tc[2]
            -1.37567212e-13 * tc[3];
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
    }

    /*species with midpoint at T=1382 kelvin */
    if (T < 1382) {
        /*species 30: C7H15X2 */
        species[30] =
            +7.56726570e-02
            -8.14947268e-05 * tc[1]
            +2.79803683e-08 * tc[2]
            -1.96944298e-12 * tc[3];
    } else {
        /*species 30: C7H15X2 */
        species[30] =
            +3.23324804e-02
            -2.18547614e-05 * tc[1]
            +5.05071180e-09 * tc[2]
            -3.88709636e-13 * tc[3];
    }

    /*species with midpoint at T=1391 kelvin */
    if (T < 1391) {
        /*species 31: NXC7H16 */
        species[31] =
            +8.54355820e-02
            -1.05069357e-04 * tc[1]
            +4.88837163e-08 * tc[2]
            -8.09579700e-12 * tc[3];
    } else {
        /*species 31: NXC7H16 */
        species[31] =
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
    double g_RT_qss[20];
    gibbs_qss(g_RT_qss, tc);

    /*reaction 1: H + O2 (+M) => HO2 (+M) */
    kc[0] = 1.0 / (refC) * exp((g_RT[3] + g_RT[6]) - (g_RT[7]));

    /*reaction 2: H2O2 (+M) => 2.000000 OH (+M) */
    kc[1] = refC * exp((g_RT[8]) - (2.000000 * g_RT[4]));

    /*reaction 3: OH + CH3 (+M) => CH3OH (+M) */
    kc[2] = 1.0 / (refC) * exp((g_RT[4] + g_RT[13]) - (g_RT[15]));

    /*reaction 4: CH3 + H (+M) => CH4 (+M) */
    kc[3] = 1.0 / (refC) * exp((g_RT[13] + g_RT[3]) - (g_RT[14]));

    /*reaction 5: 2.000000 CH3 (+M) => C2H6 (+M) */
    kc[4] = 1.0 / (refC) * exp((2.000000 * g_RT[13]) - (g_RT[16]));

    /*reaction 6: CO + CH2 (+M) => CH2CO (+M) */
    kc[5] = 1.0 / (refC) * exp((g_RT[10] + g_RT_qss[2]) - (g_RT[17]));

    /*reaction 7: CO + O (+M) => CO2 (+M) */
    kc[6] = 1.0 / (refC) * exp((g_RT[10] + g_RT[1]) - (g_RT[9]));

    /*reaction 8: CH3O (+M) => CH2O + H (+M) */
    kc[7] = refC * exp((g_RT_qss[3]) - (g_RT[11] + g_RT[3]));

    /*reaction 9: C2H3 (+M) => H + C2H2 (+M) */
    kc[8] = refC * exp((g_RT_qss[7]) - (g_RT[3] + g_RT[20]));

    /*reaction 10: H + C2H4 (+M) <=> C2H5 (+M) */
    kc[9] = 1.0 / (refC) * exp((g_RT[3] + g_RT[22]) - (g_RT_qss[4]));

    /*reaction 11: CH3CO (+M) => CH3 + CO (+M) */
    kc[10] = refC * exp((g_RT[23]) - (g_RT[13] + g_RT[10]));

    /*reaction 12: H + OH + M => H2O + M */
    kc[11] = 1.0 / (refC) * exp((g_RT[3] + g_RT[4]) - (g_RT[5]));

    /*reaction 13: CH2GSG + M => CH2 + M */
    kc[12] = exp((g_RT[12]) - (g_RT_qss[2]));

    /*reaction 14: CH2 + M => CH2GSG + M */
    kc[13] = exp((g_RT_qss[2]) - (g_RT[12]));

    /*reaction 15: HCO + M => H + CO + M */
    kc[14] = refC * exp((g_RT_qss[1]) - (g_RT[3] + g_RT[10]));

    /*reaction 16: CH3O2 + M => CH3 + O2 + M */
    kc[15] = refC * exp((g_RT[19]) - (g_RT[13] + g_RT[6]));

    /*reaction 17: CH3 + O2 + M => CH3O2 + M */
    kc[16] = 1.0 / (refC) * exp((g_RT[13] + g_RT[6]) - (g_RT[19]));

    /*reaction 18: C2H5O + M => CH3 + CH2O + M */
    kc[17] = refC * exp((g_RT_qss[9]) - (g_RT[13] + g_RT[11]));

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
    kc[43] = exp((g_RT[12] + g_RT[3]) - (g_RT_qss[0] + g_RT[2]));

    /*reaction 45: CH2GSG + H2 => CH3 + H */
    kc[44] = exp((g_RT[12] + g_RT[2]) - (g_RT[13] + g_RT[3]));

    /*reaction 46: CH3 + H => CH2GSG + H2 */
    kc[45] = exp((g_RT[13] + g_RT[3]) - (g_RT[12] + g_RT[2]));

    /*reaction 47: CH2GSG + O2 => CO + OH + H */
    kc[46] = refC * exp((g_RT[12] + g_RT[6]) - (g_RT[10] + g_RT[4] + g_RT[3]));

    /*reaction 48: CH2GSG + OH => CH2O + H */
    kc[47] = exp((g_RT[12] + g_RT[4]) - (g_RT[11] + g_RT[3]));

    /*reaction 49: CH3 + OH => CH2O + H2 */
    kc[48] = exp((g_RT[13] + g_RT[4]) - (g_RT[11] + g_RT[2]));

    /*reaction 50: CH3 + OH => CH2GSG + H2O */
    kc[49] = exp((g_RT[13] + g_RT[4]) - (g_RT[12] + g_RT[5]));

    /*reaction 51: CH2GSG + H2O => CH3 + OH */
    kc[50] = exp((g_RT[12] + g_RT[5]) - (g_RT[13] + g_RT[4]));

    /*reaction 52: CH3 + O => CH2O + H */
    kc[51] = exp((g_RT[13] + g_RT[1]) - (g_RT[11] + g_RT[3]));

    /*reaction 53: CH3 + HO2 => CH3O + OH */
    kc[52] = exp((g_RT[13] + g_RT[7]) - (g_RT_qss[3] + g_RT[4]));

    /*reaction 54: CH3 + HO2 => CH4 + O2 */
    kc[53] = exp((g_RT[13] + g_RT[7]) - (g_RT[14] + g_RT[6]));

    /*reaction 55: CH3 + O2 => CH2O + OH */
    kc[54] = exp((g_RT[13] + g_RT[6]) - (g_RT[11] + g_RT[4]));

    /*reaction 56: CH3 + H => CH2 + H2 */
    kc[55] = exp((g_RT[13] + g_RT[3]) - (g_RT_qss[2] + g_RT[2]));

    /*reaction 57: CH2 + H2 => CH3 + H */
    kc[56] = exp((g_RT_qss[2] + g_RT[2]) - (g_RT[13] + g_RT[3]));

    /*reaction 58: 2.000000 CH3 <=> H + C2H5 */
    kc[57] = exp((2.000000 * g_RT[13]) - (g_RT[3] + g_RT_qss[4]));

    /*reaction 59: CH3 + OH => CH2 + H2O */
    kc[58] = exp((g_RT[13] + g_RT[4]) - (g_RT_qss[2] + g_RT[5]));

    /*reaction 60: CH2 + H2O => CH3 + OH */
    kc[59] = exp((g_RT_qss[2] + g_RT[5]) - (g_RT[13] + g_RT[4]));

    /*reaction 61: CH4 + O => CH3 + OH */
    kc[60] = exp((g_RT[14] + g_RT[1]) - (g_RT[13] + g_RT[4]));

    /*reaction 62: CH4 + H => CH3 + H2 */
    kc[61] = exp((g_RT[14] + g_RT[3]) - (g_RT[13] + g_RT[2]));

    /*reaction 63: CH3 + H2 => CH4 + H */
    kc[62] = exp((g_RT[13] + g_RT[2]) - (g_RT[14] + g_RT[3]));

    /*reaction 64: CH4 + OH => CH3 + H2O */
    kc[63] = exp((g_RT[14] + g_RT[4]) - (g_RT[13] + g_RT[5]));

    /*reaction 65: CH3 + H2O => CH4 + OH */
    kc[64] = exp((g_RT[13] + g_RT[5]) - (g_RT[14] + g_RT[4]));

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
    kc[72] = exp((g_RT_qss[1] + g_RT[13]) - (g_RT[14] + g_RT[10]));

    /*reaction 74: CH2O + OH => HCO + H2O */
    kc[73] = exp((g_RT[11] + g_RT[4]) - (g_RT_qss[1] + g_RT[5]));

    /*reaction 75: CH2O + O => HCO + OH */
    kc[74] = exp((g_RT[11] + g_RT[1]) - (g_RT_qss[1] + g_RT[4]));

    /*reaction 76: CH2O + H => HCO + H2 */
    kc[75] = exp((g_RT[11] + g_RT[3]) - (g_RT_qss[1] + g_RT[2]));

    /*reaction 77: CH2O + CH3 => HCO + CH4 */
    kc[76] = exp((g_RT[11] + g_RT[13]) - (g_RT_qss[1] + g_RT[14]));

    /*reaction 78: 2.000000 CH3O => CH3OH + CH2O */
    kc[77] = exp((2.000000 * g_RT_qss[3]) - (g_RT[15] + g_RT[11]));

    /*reaction 79: CH3O + O2 => CH2O + HO2 */
    kc[78] = exp((g_RT_qss[3] + g_RT[6]) - (g_RT[11] + g_RT[7]));

    /*reaction 80: CH3O + H2 => CH3OH + H */
    kc[79] = exp((g_RT_qss[3] + g_RT[2]) - (g_RT[15] + g_RT[3]));

    /*reaction 81: CH3OH + OH => CH3O + H2O */
    kc[80] = exp((g_RT[15] + g_RT[4]) - (g_RT_qss[3] + g_RT[5]));

    /*reaction 82: CH2GSG + CO2 => CH2O + CO */
    kc[81] = exp((g_RT[12] + g_RT[9]) - (g_RT[11] + g_RT[10]));

    /*reaction 83: HOCHO + H => H2 + CO + OH */
    kc[82] = refC * exp((g_RT[18] + g_RT[3]) - (g_RT[2] + g_RT[10] + g_RT[4]));

    /*reaction 84: HOCHO + OH => H2O + CO + OH */
    kc[83] = refC * exp((g_RT[18] + g_RT[4]) - (g_RT[5] + g_RT[10] + g_RT[4]));

    /*reaction 85: HOCHO => HCO + OH */
    kc[84] = refC * exp((g_RT[18]) - (g_RT_qss[1] + g_RT[4]));

    /*reaction 86: HCO + OH => HOCHO */
    kc[85] = 1.0 / (refC) * exp((g_RT_qss[1] + g_RT[4]) - (g_RT[18]));

    /*reaction 87: HOCHO + H => H2 + CO2 + H */
    kc[86] = refC * exp((g_RT[18] + g_RT[3]) - (g_RT[2] + g_RT[9] + g_RT[3]));

    /*reaction 88: HOCHO + OH => H2O + CO2 + H */
    kc[87] = refC * exp((g_RT[18] + g_RT[4]) - (g_RT[5] + g_RT[9] + g_RT[3]));

    /*reaction 89: 2.000000 CH3O2 => O2 + 2.000000 CH3O */
    kc[88] = refC * exp((2.000000 * g_RT[19]) - (g_RT[6] + 2.000000 * g_RT_qss[3]));

    /*reaction 90: CH3O2 + CH3 => 2.000000 CH3O */
    kc[89] = exp((g_RT[19] + g_RT[13]) - (2.000000 * g_RT_qss[3]));

    /*reaction 91: 2.000000 CH3O2 => CH2O + CH3OH + O2 */
    kc[90] = refC * exp((2.000000 * g_RT[19]) - (g_RT[11] + g_RT[15] + g_RT[6]));

    /*reaction 92: CH3O2 + HO2 => CH3O2H + O2 */
    kc[91] = exp((g_RT[19] + g_RT[7]) - (g_RT_qss[5] + g_RT[6]));

    /*reaction 93: CH3O2H => CH3O + OH */
    kc[92] = refC * exp((g_RT_qss[5]) - (g_RT_qss[3] + g_RT[4]));

    /*reaction 94: C2H2 + O => CH2 + CO */
    kc[93] = exp((g_RT[20] + g_RT[1]) - (g_RT_qss[2] + g_RT[10]));

    /*reaction 95: C2H2 + O => HCCO + H */
    kc[94] = exp((g_RT[20] + g_RT[1]) - (g_RT_qss[6] + g_RT[3]));

    /*reaction 96: C2H3 + H => C2H2 + H2 */
    kc[95] = exp((g_RT_qss[7] + g_RT[3]) - (g_RT[20] + g_RT[2]));

    /*reaction 97: C2H3 + O2 => CH2CHO + O */
    kc[96] = exp((g_RT_qss[7] + g_RT[6]) - (g_RT_qss[8] + g_RT[1]));

    /*reaction 98: C2H3 + CH3 => C3H6 */
    kc[97] = 1.0 / (refC) * exp((g_RT_qss[7] + g_RT[13]) - (g_RT[21]));

    /*reaction 99: C2H3 + O2 => C2H2 + HO2 */
    kc[98] = exp((g_RT_qss[7] + g_RT[6]) - (g_RT[20] + g_RT[7]));

    /*reaction 100: C2H3 + O2 => CH2O + HCO */
    kc[99] = exp((g_RT_qss[7] + g_RT[6]) - (g_RT[11] + g_RT_qss[1]));

    /*reaction 101: C2H4 + CH3 => C2H3 + CH4 */
    kc[100] = exp((g_RT[22] + g_RT[13]) - (g_RT_qss[7] + g_RT[14]));

    /*reaction 102: C2H4 + O => CH3 + HCO */
    kc[101] = exp((g_RT[22] + g_RT[1]) - (g_RT[13] + g_RT_qss[1]));

    /*reaction 103: C2H4 + OH => C2H3 + H2O */
    kc[102] = exp((g_RT[22] + g_RT[4]) - (g_RT_qss[7] + g_RT[5]));

    /*reaction 104: C2H4 + O => CH2CHO + H */
    kc[103] = exp((g_RT[22] + g_RT[1]) - (g_RT_qss[8] + g_RT[3]));

    /*reaction 105: C2H4 + H => C2H3 + H2 */
    kc[104] = exp((g_RT[22] + g_RT[3]) - (g_RT_qss[7] + g_RT[2]));

    /*reaction 106: C2H3 + H2 => C2H4 + H */
    kc[105] = exp((g_RT_qss[7] + g_RT[2]) - (g_RT[22] + g_RT[3]));

    /*reaction 107: H + C2H5 => C2H6 */
    kc[106] = 1.0 / (refC) * exp((g_RT[3] + g_RT_qss[4]) - (g_RT[16]));

    /*reaction 108: CH3O2 + C2H5 => CH3O + C2H5O */
    kc[107] = exp((g_RT[19] + g_RT_qss[4]) - (g_RT_qss[3] + g_RT_qss[9]));

    /*reaction 109: C2H5 + HO2 => C2H5O + OH */
    kc[108] = exp((g_RT_qss[4] + g_RT[7]) - (g_RT_qss[9] + g_RT[4]));

    /*reaction 110: C2H5 + O2 => C2H4 + HO2 */
    kc[109] = exp((g_RT_qss[4] + g_RT[6]) - (g_RT[22] + g_RT[7]));

    /*reaction 111: C2H6 + O => C2H5 + OH */
    kc[110] = exp((g_RT[16] + g_RT[1]) - (g_RT_qss[4] + g_RT[4]));

    /*reaction 112: C2H6 + OH => C2H5 + H2O */
    kc[111] = exp((g_RT[16] + g_RT[4]) - (g_RT_qss[4] + g_RT[5]));

    /*reaction 113: C2H6 + H => C2H5 + H2 */
    kc[112] = exp((g_RT[16] + g_RT[3]) - (g_RT_qss[4] + g_RT[2]));

    /*reaction 114: HCCO + O => H + 2.000000 CO */
    kc[113] = refC * exp((g_RT_qss[6] + g_RT[1]) - (g_RT[3] + 2.000000 * g_RT[10]));

    /*reaction 115: HCCO + OH => 2.000000 HCO */
    kc[114] = exp((g_RT_qss[6] + g_RT[4]) - (2.000000 * g_RT_qss[1]));

    /*reaction 116: HCCO + O2 => CO2 + HCO */
    kc[115] = exp((g_RT_qss[6] + g_RT[6]) - (g_RT[9] + g_RT_qss[1]));

    /*reaction 117: HCCO + H => CH2GSG + CO */
    kc[116] = exp((g_RT_qss[6] + g_RT[3]) - (g_RT[12] + g_RT[10]));

    /*reaction 118: CH2GSG + CO => HCCO + H */
    kc[117] = exp((g_RT[12] + g_RT[10]) - (g_RT_qss[6] + g_RT[3]));

    /*reaction 119: CH2CO + O => HCCO + OH */
    kc[118] = exp((g_RT[17] + g_RT[1]) - (g_RT_qss[6] + g_RT[4]));

    /*reaction 120: CH2CO + H => HCCO + H2 */
    kc[119] = exp((g_RT[17] + g_RT[3]) - (g_RT_qss[6] + g_RT[2]));

    /*reaction 121: HCCO + H2 => CH2CO + H */
    kc[120] = exp((g_RT_qss[6] + g_RT[2]) - (g_RT[17] + g_RT[3]));

    /*reaction 122: CH2CO + H => CH3 + CO */
    kc[121] = exp((g_RT[17] + g_RT[3]) - (g_RT[13] + g_RT[10]));

    /*reaction 123: CH2CO + O => CH2 + CO2 */
    kc[122] = exp((g_RT[17] + g_RT[1]) - (g_RT_qss[2] + g_RT[9]));

    /*reaction 124: CH2CO + OH => HCCO + H2O */
    kc[123] = exp((g_RT[17] + g_RT[4]) - (g_RT_qss[6] + g_RT[5]));

    /*reaction 125: CH2CHO + O2 => CH2O + CO + OH */
    kc[124] = refC * exp((g_RT_qss[8] + g_RT[6]) - (g_RT[11] + g_RT[10] + g_RT[4]));

    /*reaction 126: CH2CHO => CH2CO + H */
    kc[125] = refC * exp((g_RT_qss[8]) - (g_RT[17] + g_RT[3]));

    /*reaction 127: CH2CO + H => CH2CHO */
    kc[126] = 1.0 / (refC) * exp((g_RT[17] + g_RT[3]) - (g_RT_qss[8]));

    /*reaction 128: C2H5O2 => C2H5 + O2 */
    kc[127] = refC * exp((g_RT_qss[10]) - (g_RT_qss[4] + g_RT[6]));

    /*reaction 129: C2H5 + O2 => C2H5O2 */
    kc[128] = 1.0 / (refC) * exp((g_RT_qss[4] + g_RT[6]) - (g_RT_qss[10]));

    /*reaction 130: C2H5O2 => C2H4 + HO2 */
    kc[129] = refC * exp((g_RT_qss[10]) - (g_RT[22] + g_RT[7]));

    /*reaction 131: C3H2 + O2 => HCCO + CO + H */
    kc[130] = refC * exp((g_RT_qss[11] + g_RT[6]) - (g_RT_qss[6] + g_RT[10] + g_RT[3]));

    /*reaction 132: C3H2 + OH => C2H2 + HCO */
    kc[131] = exp((g_RT_qss[11] + g_RT[4]) - (g_RT[20] + g_RT_qss[1]));

    /*reaction 133: C3H3 + O2 => CH2CO + HCO */
    kc[132] = exp((g_RT_qss[12] + g_RT[6]) - (g_RT[17] + g_RT_qss[1]));

    /*reaction 134: C3H3 + HO2 => C3H4XA + O2 */
    kc[133] = exp((g_RT_qss[12] + g_RT[7]) - (g_RT[24] + g_RT[6]));

    /*reaction 135: C3H3 + H => C3H2 + H2 */
    kc[134] = exp((g_RT_qss[12] + g_RT[3]) - (g_RT_qss[11] + g_RT[2]));

    /*reaction 136: C3H3 + OH => C3H2 + H2O */
    kc[135] = exp((g_RT_qss[12] + g_RT[4]) - (g_RT_qss[11] + g_RT[5]));

    /*reaction 137: C3H2 + H2O => C3H3 + OH */
    kc[136] = exp((g_RT_qss[11] + g_RT[5]) - (g_RT_qss[12] + g_RT[4]));

    /*reaction 138: C3H4XA + H => C3H3 + H2 */
    kc[137] = exp((g_RT[24] + g_RT[3]) - (g_RT_qss[12] + g_RT[2]));

    /*reaction 139: C3H4XA + OH => C3H3 + H2O */
    kc[138] = exp((g_RT[24] + g_RT[4]) - (g_RT_qss[12] + g_RT[5]));

    /*reaction 140: C3H4XA + O => C2H4 + CO */
    kc[139] = exp((g_RT[24] + g_RT[1]) - (g_RT[22] + g_RT[10]));

    /*reaction 141: C3H5XA + H => C3H4XA + H2 */
    kc[140] = exp((g_RT[25] + g_RT[3]) - (g_RT[24] + g_RT[2]));

    /*reaction 142: C3H5XA + HO2 => C3H6 + O2 */
    kc[141] = exp((g_RT[25] + g_RT[7]) - (g_RT[21] + g_RT[6]));

    /*reaction 143: C3H5XA + H => C3H6 */
    kc[142] = 1.0 / (refC) * exp((g_RT[25] + g_RT[3]) - (g_RT[21]));

    /*reaction 144: C3H5XA => C2H2 + CH3 */
    kc[143] = refC * exp((g_RT[25]) - (g_RT[20] + g_RT[13]));

    /*reaction 145: C3H5XA => C3H4XA + H */
    kc[144] = refC * exp((g_RT[25]) - (g_RT[24] + g_RT[3]));

    /*reaction 146: C3H4XA + H => C3H5XA */
    kc[145] = 1.0 / (refC) * exp((g_RT[24] + g_RT[3]) - (g_RT[25]));

    /*reaction 147: C3H5XA + CH2O => C3H6 + HCO */
    kc[146] = exp((g_RT[25] + g_RT[11]) - (g_RT[21] + g_RT_qss[1]));

    /*reaction 148: 2.000000 C3H5XA => C3H4XA + C3H6 */
    kc[147] = exp((2.000000 * g_RT[25]) - (g_RT[24] + g_RT[21]));

    /*reaction 149: C3H6 + H => C2H4 + CH3 */
    kc[148] = exp((g_RT[21] + g_RT[3]) - (g_RT[22] + g_RT[13]));

    /*reaction 150: C3H6 + H => C3H5XA + H2 */
    kc[149] = exp((g_RT[21] + g_RT[3]) - (g_RT[25] + g_RT[2]));

    /*reaction 151: C3H6 + O => C2H5 + HCO */
    kc[150] = exp((g_RT[21] + g_RT[1]) - (g_RT_qss[4] + g_RT_qss[1]));

    /*reaction 152: C3H6 + O => C3H5XA + OH */
    kc[151] = exp((g_RT[21] + g_RT[1]) - (g_RT[25] + g_RT[4]));

    /*reaction 153: C3H6 + O => CH2CO + CH3 + H */
    kc[152] = refC * exp((g_RT[21] + g_RT[1]) - (g_RT[17] + g_RT[13] + g_RT[3]));

    /*reaction 154: C3H6 + OH => C3H5XA + H2O */
    kc[153] = exp((g_RT[21] + g_RT[4]) - (g_RT[25] + g_RT[5]));

    /*reaction 155: NXC3H7 + O2 => C3H6 + HO2 */
    kc[154] = exp((g_RT_qss[13] + g_RT[6]) - (g_RT[21] + g_RT[7]));

    /*reaction 156: NXC3H7 => CH3 + C2H4 */
    kc[155] = refC * exp((g_RT_qss[13]) - (g_RT[13] + g_RT[22]));

    /*reaction 157: CH3 + C2H4 => NXC3H7 */
    kc[156] = 1.0 / (refC) * exp((g_RT[13] + g_RT[22]) - (g_RT_qss[13]));

    /*reaction 158: NXC3H7 => H + C3H6 */
    kc[157] = refC * exp((g_RT_qss[13]) - (g_RT[3] + g_RT[21]));

    /*reaction 159: H + C3H6 => NXC3H7 */
    kc[158] = 1.0 / (refC) * exp((g_RT[3] + g_RT[21]) - (g_RT_qss[13]));

    /*reaction 160: NXC3H7O2 => NXC3H7 + O2 */
    kc[159] = refC * exp((g_RT[26]) - (g_RT_qss[13] + g_RT[6]));

    /*reaction 161: NXC3H7 + O2 => NXC3H7O2 */
    kc[160] = 1.0 / (refC) * exp((g_RT_qss[13] + g_RT[6]) - (g_RT[26]));

    /*reaction 162: C4H6 => 2.000000 C2H3 */
    kc[161] = refC * exp((g_RT[27]) - (2.000000 * g_RT_qss[7]));

    /*reaction 163: 2.000000 C2H3 => C4H6 */
    kc[162] = 1.0 / (refC) * exp((2.000000 * g_RT_qss[7]) - (g_RT[27]));

    /*reaction 164: C4H6 + OH => CH2O + C3H5XA */
    kc[163] = exp((g_RT[27] + g_RT[4]) - (g_RT[11] + g_RT[25]));

    /*reaction 165: C4H6 + OH => C2H5 + CH2CO */
    kc[164] = exp((g_RT[27] + g_RT[4]) - (g_RT_qss[4] + g_RT[17]));

    /*reaction 166: C4H6 + O => C2H4 + CH2CO */
    kc[165] = exp((g_RT[27] + g_RT[1]) - (g_RT[22] + g_RT[17]));

    /*reaction 167: C4H6 + H => C2H3 + C2H4 */
    kc[166] = exp((g_RT[27] + g_RT[3]) - (g_RT_qss[7] + g_RT[22]));

    /*reaction 168: C4H6 + O => CH2O + C3H4XA */
    kc[167] = exp((g_RT[27] + g_RT[1]) - (g_RT[11] + g_RT[24]));

    /*reaction 169: H + C4H7 => C4H8X1 */
    kc[168] = 1.0 / (refC) * exp((g_RT[3] + g_RT[28]) - (g_RT[29]));

    /*reaction 170: C3H5XA + C4H7 => C3H6 + C4H6 */
    kc[169] = exp((g_RT[25] + g_RT[28]) - (g_RT[21] + g_RT[27]));

    /*reaction 171: C2H5 + C4H7 => C4H6 + C2H6 */
    kc[170] = exp((g_RT_qss[4] + g_RT[28]) - (g_RT[27] + g_RT[16]));

    /*reaction 172: C4H7 => C4H6 + H */
    kc[171] = refC * exp((g_RT[28]) - (g_RT[27] + g_RT[3]));

    /*reaction 173: C4H6 + H => C4H7 */
    kc[172] = 1.0 / (refC) * exp((g_RT[27] + g_RT[3]) - (g_RT[28]));

    /*reaction 174: C4H7 + CH3 => C4H6 + CH4 */
    kc[173] = exp((g_RT[28] + g_RT[13]) - (g_RT[27] + g_RT[14]));

    /*reaction 175: C4H7 + HO2 => C4H8X1 + O2 */
    kc[174] = exp((g_RT[28] + g_RT[7]) - (g_RT[29] + g_RT[6]));

    /*reaction 176: C4H7 + O2 => C4H6 + HO2 */
    kc[175] = exp((g_RT[28] + g_RT[6]) - (g_RT[27] + g_RT[7]));

    /*reaction 177: C4H7 => C2H4 + C2H3 */
    kc[176] = refC * exp((g_RT[28]) - (g_RT[22] + g_RT_qss[7]));

    /*reaction 178: H + C4H7 => C4H6 + H2 */
    kc[177] = exp((g_RT[3] + g_RT[28]) - (g_RT[27] + g_RT[2]));

    /*reaction 179: C4H8X1 + H => C4H7 + H2 */
    kc[178] = exp((g_RT[29] + g_RT[3]) - (g_RT[28] + g_RT[2]));

    /*reaction 180: C4H8X1 + OH => NXC3H7 + CH2O */
    kc[179] = exp((g_RT[29] + g_RT[4]) - (g_RT_qss[13] + g_RT[11]));

    /*reaction 181: C4H8X1 + OH => CH3CO + C2H6 */
    kc[180] = exp((g_RT[29] + g_RT[4]) - (g_RT[23] + g_RT[16]));

    /*reaction 182: C4H8X1 + O => CH3CO + C2H5 */
    kc[181] = exp((g_RT[29] + g_RT[1]) - (g_RT[23] + g_RT_qss[4]));

    /*reaction 183: C4H8X1 + O => C3H6 + CH2O */
    kc[182] = exp((g_RT[29] + g_RT[1]) - (g_RT[21] + g_RT[11]));

    /*reaction 184: C4H8X1 + OH => C4H7 + H2O */
    kc[183] = exp((g_RT[29] + g_RT[4]) - (g_RT[28] + g_RT[5]));

    /*reaction 185: C4H8X1 => C3H5XA + CH3 */
    kc[184] = refC * exp((g_RT[29]) - (g_RT[25] + g_RT[13]));

    /*reaction 186: C3H5XA + CH3 => C4H8X1 */
    kc[185] = 1.0 / (refC) * exp((g_RT[25] + g_RT[13]) - (g_RT[29]));

    /*reaction 187: PXC4H9 => C4H8X1 + H */
    kc[186] = refC * exp((g_RT_qss[14]) - (g_RT[29] + g_RT[3]));

    /*reaction 188: C4H8X1 + H => PXC4H9 */
    kc[187] = 1.0 / (refC) * exp((g_RT[29] + g_RT[3]) - (g_RT_qss[14]));

    /*reaction 189: PXC4H9 => C2H5 + C2H4 */
    kc[188] = refC * exp((g_RT_qss[14]) - (g_RT_qss[4] + g_RT[22]));

    /*reaction 190: PXC4H9O2 => PXC4H9 + O2 */
    kc[189] = refC * exp((g_RT_qss[15]) - (g_RT_qss[14] + g_RT[6]));

    /*reaction 191: PXC4H9 + O2 => PXC4H9O2 */
    kc[190] = 1.0 / (refC) * exp((g_RT_qss[14] + g_RT[6]) - (g_RT_qss[15]));

    /*reaction 192: C5H9 => C4H6 + CH3 */
    kc[191] = refC * exp((g_RT_qss[16]) - (g_RT[27] + g_RT[13]));

    /*reaction 193: C5H9 => C3H5XA + C2H4 */
    kc[192] = refC * exp((g_RT_qss[16]) - (g_RT[25] + g_RT[22]));

    /*reaction 194: C5H10X1 + OH => C5H9 + H2O */
    kc[193] = exp((g_RT_qss[17] + g_RT[4]) - (g_RT_qss[16] + g_RT[5]));

    /*reaction 195: C5H10X1 + H => C5H9 + H2 */
    kc[194] = exp((g_RT_qss[17] + g_RT[3]) - (g_RT_qss[16] + g_RT[2]));

    /*reaction 196: C5H10X1 => C2H5 + C3H5XA */
    kc[195] = refC * exp((g_RT_qss[17]) - (g_RT_qss[4] + g_RT[25]));

    /*reaction 197: C2H5 + C3H5XA => C5H10X1 */
    kc[196] = 1.0 / (refC) * exp((g_RT_qss[4] + g_RT[25]) - (g_RT_qss[17]));

    /*reaction 198: C5H10X1 + O => C5H9 + OH */
    kc[197] = exp((g_RT_qss[17] + g_RT[1]) - (g_RT_qss[16] + g_RT[4]));

    /*reaction 199: C5H11X1 => C3H6 + C2H5 */
    kc[198] = refC * exp((g_RT_qss[18]) - (g_RT[21] + g_RT_qss[4]));

    /*reaction 200: C5H11X1 => C2H4 + NXC3H7 */
    kc[199] = refC * exp((g_RT_qss[18]) - (g_RT[22] + g_RT_qss[13]));

    /*reaction 201: C5H11X1 <=> C5H10X1 + H */
    kc[200] = refC * exp((g_RT_qss[18]) - (g_RT_qss[17] + g_RT[3]));

    /*reaction 202: C6H12X1 => NXC3H7 + C3H5XA */
    kc[201] = refC * exp((g_RT_qss[19]) - (g_RT_qss[13] + g_RT[25]));

    /*reaction 203: C6H12X1 + OH => C5H11X1 + CH2O */
    kc[202] = exp((g_RT_qss[19] + g_RT[4]) - (g_RT_qss[18] + g_RT[11]));

    /*reaction 204: C7H15X2 => C6H12X1 + CH3 */
    kc[203] = refC * exp((g_RT[30]) - (g_RT_qss[19] + g_RT[13]));

    /*reaction 205: C7H15X2 => PXC4H9 + C3H6 */
    kc[204] = refC * exp((g_RT[30]) - (g_RT_qss[14] + g_RT[21]));

    /*reaction 206: C7H15X2 => C4H8X1 + NXC3H7 */
    kc[205] = refC * exp((g_RT[30]) - (g_RT[29] + g_RT_qss[13]));

    /*reaction 207: C7H15X2 => C5H11X1 + C2H4 */
    kc[206] = refC * exp((g_RT[30]) - (g_RT_qss[18] + g_RT[22]));

    /*reaction 208: C7H15X2 => C2H5 + C5H10X1 */
    kc[207] = refC * exp((g_RT[30]) - (g_RT_qss[4] + g_RT_qss[17]));

    /*reaction 209: C7H15X2 + HO2 => NXC7H16 + O2 */
    kc[208] = exp((g_RT[30] + g_RT[7]) - (g_RT[31] + g_RT[6]));

    /*reaction 210: NXC7H16 + CH3O2 => C7H15X2 + CH3O2H */
    kc[209] = exp((g_RT[31] + g_RT[19]) - (g_RT[30] + g_RT_qss[5]));

    /*reaction 211: NXC7H16 + H => C7H15X2 + H2 */
    kc[210] = exp((g_RT[31] + g_RT[3]) - (g_RT[30] + g_RT[2]));

    /*reaction 212: NXC7H16 => PXC4H9 + NXC3H7 */
    kc[211] = refC * exp((g_RT[31]) - (g_RT_qss[14] + g_RT_qss[13]));

    /*reaction 213: NXC7H16 + HO2 => C7H15X2 + H2O2 */
    kc[212] = exp((g_RT[31] + g_RT[7]) - (g_RT[30] + g_RT[8]));

    /*reaction 214: NXC7H16 => C5H11X1 + C2H5 */
    kc[213] = refC * exp((g_RT[31]) - (g_RT_qss[18] + g_RT_qss[4]));

    /*reaction 215: NXC7H16 + CH3O => C7H15X2 + CH3OH */
    kc[214] = exp((g_RT[31] + g_RT_qss[3]) - (g_RT[30] + g_RT[15]));

    /*reaction 216: NXC7H16 + O => C7H15X2 + OH */
    kc[215] = exp((g_RT[31] + g_RT[1]) - (g_RT[30] + g_RT[4]));

    /*reaction 217: NXC7H16 + OH => C7H15X2 + H2O */
    kc[216] = exp((g_RT[31] + g_RT[4]) - (g_RT[30] + g_RT[5]));

    /*reaction 218: NXC7H16 + CH3 => C7H15X2 + CH4 */
    kc[217] = exp((g_RT[31] + g_RT[13]) - (g_RT[30] + g_RT[14]));

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
        /*species 12: CH2GSG */
        species[12] =
            +4.989368000000000e+04 * invT
            +3.913732930000000e+00
            -3.971265000000000e+00 * tc[0]
            +8.495445000000000e-05 * tc[1]
            -1.708948333333333e-07 * tc[2]
            -2.077125833333333e-10 * tc[3]
            +9.906330000000000e-14 * tc[4];
        /*species 13: CH3 */
        species[13] =
            +1.642378000000000e+04 * invT
            -4.359351000000000e+00
            -2.430443000000000e+00 * tc[0]
            -5.562050000000000e-03 * tc[1]
            +2.800366666666666e-06 * tc[2]
            -1.351524166666667e-09 * tc[3]
            +2.932476500000000e-13 * tc[4];
        /*species 14: CH4 */
        species[14] =
            -9.825228999999999e+03 * invT
            -1.294344850000000e+01
            -7.787415000000000e-01 * tc[0]
            -8.738340000000001e-03 * tc[1]
            +4.639015000000000e-06 * tc[2]
            -2.541423333333333e-09 * tc[3]
            +6.119655000000000e-13 * tc[4];
        /*species 15: CH3OH */
        species[15] =
            -2.535348000000000e+04 * invT
            -8.572515000000001e+00
            -2.660115000000000e+00 * tc[0]
            -3.670754000000000e-03 * tc[1]
            -1.195008500000000e-06 * tc[2]
            +7.327661666666667e-10 * tc[3]
            -1.195285000000000e-13 * tc[4];
        /*species 17: CH2CO */
        species[17] =
            -7.632637000000000e+03 * invT
            -5.698582000000000e+00
            -2.974971000000000e+00 * tc[0]
            -6.059355000000000e-03 * tc[1]
            +3.908410000000000e-07 * tc[2]
            +5.388904166666666e-10 * tc[3]
            -1.952824500000000e-13 * tc[4];
        /*species 20: C2H2 */
        species[20] =
            +2.612444000000000e+04 * invT
            -6.791815999999999e+00
            -2.013562000000000e+00 * tc[0]
            -7.595225000000000e-03 * tc[1]
            +2.693865000000000e-06 * tc[2]
            -7.565826666666667e-10 * tc[3]
            +9.563730000000000e-14 * tc[4];
        /*species 22: C2H4 */
        species[22] =
            +5.573046000000000e+03 * invT
            -2.507297800000000e+01
            +8.614880000000000e-01 * tc[0]
            -1.398081500000000e-02 * tc[1]
            +5.647795000000000e-06 * tc[2]
            -2.320960000000000e-09 * tc[3]
            +4.868939500000000e-13 * tc[4];
        /*species 23: CH3CO */
        species[23] =
            -4.108508000000000e+03 * invT
            -8.103572000000000e+00
            -3.125278000000000e+00 * tc[0]
            -4.889110000000000e-03 * tc[1]
            -7.535746666666667e-07 * tc[2]
            +7.507885000000000e-10 * tc[3]
            -1.596859000000000e-13 * tc[4];
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
        /*species 12: CH2GSG */
        species[12] =
            +4.984975000000000e+04 * invT
            +1.866319000000000e+00
            -3.552889000000000e+00 * tc[0]
            -1.033394000000000e-03 * tc[1]
            +3.190193333333333e-08 * tc[2]
            +9.205608333333333e-12 * tc[3]
            -1.010675000000000e-15 * tc[4];
        /*species 13: CH3 */
        species[13] =
            +1.643781000000000e+04 * invT
            -2.608645000000000e+00
            -2.844052000000000e+00 * tc[0]
            -3.068987000000000e-03 * tc[1]
            +3.717241666666666e-07 * tc[2]
            -3.154300833333333e-11 * tc[3]
            +1.226079500000000e-15 * tc[4];
        /*species 14: CH4 */
        species[14] =
            -1.008079000000000e+04 * invT
            -7.939916000000000e+00
            -1.683479000000000e+00 * tc[0]
            -5.118620000000000e-03 * tc[1]
            +6.458548333333333e-07 * tc[2]
            -5.654654166666667e-11 * tc[3]
            +2.251711500000000e-15 * tc[4];
        /*species 15: CH3OH */
        species[15] =
            -2.615791000000000e+04 * invT
            +1.650865000000000e+00
            -4.029061000000000e+00 * tc[0]
            -4.688296500000000e-03 * tc[1]
            +5.083756666666666e-07 * tc[2]
            -3.632327500000000e-11 * tc[3]
            +1.112361500000000e-15 * tc[4];
        /*species 17: CH2CO */
        species[17] =
            -8.583402000000000e+03 * invT
            +1.369639800000000e+01
            -6.038817000000000e+00 * tc[0]
            -2.902420000000000e-03 * tc[1]
            +3.201590000000000e-07 * tc[2]
            -2.328737500000000e-11 * tc[3]
            +7.294340000000000e-16 * tc[4];
        /*species 20: C2H2 */
        species[20] =
            +2.566766000000000e+04 * invT
            +7.237108000000000e+00
            -4.436770000000000e+00 * tc[0]
            -2.688019500000000e-03 * tc[1]
            +3.188028333333333e-07 * tc[2]
            -2.738649166666667e-11 * tc[3]
            +1.078355000000000e-15 * tc[4];
        /*species 22: C2H4 */
        species[22] =
            +4.428289000000000e+03 * invT
            +1.298030000000000e+00
            -3.528419000000000e+00 * tc[0]
            -5.742590000000000e-03 * tc[1]
            +7.363975000000000e-07 * tc[2]
            -6.537167500000001e-11 * tc[3]
            +2.633424000000000e-15 * tc[4];
        /*species 23: CH3CO */
        species[23] =
            -5.187863000000000e+03 * invT
            +8.887228000000000e+00
            -5.612279000000000e+00 * tc[0]
            -4.224943000000000e-03 * tc[1]
            +4.756911666666667e-07 * tc[2]
            -3.531980000000000e-11 * tc[3]
            +1.134202000000000e-15 * tc[4];
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
        /*species 16: C2H6 */
        species[16] =
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
        /*species 16: C2H6 */
        species[16] =
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
        /*species 18: HOCHO */
        species[18] =
            -4.646165040000000e+04 * invT
            -1.585309795000000e+01
            -1.435481850000000e+00 * tc[0]
            -8.168150799999999e-03 * tc[1]
            +1.770957016666667e-06 * tc[2]
            -2.767774808333333e-10 * tc[3]
            +2.010880515000000e-14 * tc[4];
    } else {
        /*species 18: HOCHO */
        species[18] =
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
        /*species 19: CH3O2 */
        species[19] =
            -6.843942590000000e+02 * invT
            -9.018341400000001e-01
            -4.261469060000000e+00 * tc[0]
            -5.043679950000000e-03 * tc[1]
            +5.358436400000000e-07 * tc[2]
            -1.745077225000000e-11 * tc[3]
            -2.091695515000000e-15 * tc[4];
    } else {
        /*species 19: CH3O2 */
        species[19] =
            -1.535748380000000e+03 * invT
            +1.067751777000000e+01
            -5.957878910000000e+00 * tc[0]
            -3.953643130000000e-03 * tc[1]
            +4.470770566666667e-07 * tc[2]
            -3.449094475000000e-11 * tc[3]
            +1.195036650000000e-15 * tc[4];
    }

    /*species with midpoint at T=1388 kelvin */
    if (T < 1388) {
        /*species 21: C3H6 */
        species[21] =
            +1.066881640000000e+03 * invT
            -2.150575815600000e+01
            -3.946154440000000e-01 * tc[0]
            -1.445538310000000e-02 * tc[1]
            +2.581446800000000e-06 * tc[2]
            -3.240118408333333e-10 * tc[3]
            +1.689451760000000e-14 * tc[4];
    } else {
        /*species 21: C3H6 */
        species[21] =
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
        /*species 24: C3H4XA */
        species[24] =
            +2.251243000000000e+04 * invT
            -7.395871000000000e+00
            -2.539831000000000e+00 * tc[0]
            -8.167185000000000e-03 * tc[1]
            +2.941583333333333e-07 * tc[2]
            +3.872804166666666e-10 * tc[3]
            -8.645655000000001e-14 * tc[4];
    } else {
        /*species 24: C3H4XA */
        species[24] =
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
        /*species 25: C3H5XA */
        species[25] =
            +1.938342260000000e+04 * invT
            -2.583584505800000e+01
            +5.291319580000000e-01 * tc[0]
            -1.672795500000000e-02 * tc[1]
            +4.223350450000001e-06 * tc[2]
            -8.572146166666666e-10 * tc[3]
            +8.662917000000000e-14 * tc[4];
    } else {
        /*species 25: C3H5XA */
        species[25] =
            +1.635760920000000e+04 * invT
            +3.103978458000000e+01
            -8.458839579999999e+00 * tc[0]
            -5.634774150000000e-03 * tc[1]
            +6.396547733333333e-07 * tc[2]
            -4.950492658333333e-11 * tc[3]
            +1.719590150000000e-15 * tc[4];
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
    }

    /*species with midpoint at T=1382 kelvin */
    if (T < 1382) {
        /*species 30: C7H15X2 */
        species[30] =
            -2.356053030000000e+03 * invT
            -3.377006617670000e+01
            +3.791557670000000e-02 * tc[0]
            -3.783632850000000e-02 * tc[1]
            +6.791227233333333e-06 * tc[2]
            -7.772324525000000e-10 * tc[3]
            +2.461803725000000e-14 * tc[4];
    } else {
        /*species 30: C7H15X2 */
        species[30] =
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
        /*species 31: NXC7H16 */
        species[31] =
            -2.565865650000000e+04 * invT
            -3.664165307000000e+01
            +1.268361870000000e+00 * tc[0]
            -4.271779100000000e-02 * tc[1]
            +8.755779766666667e-06 * tc[2]
            -1.357881008333333e-09 * tc[3]
            +1.011974625000000e-13 * tc[4];
    } else {
        /*species 31: NXC7H16 */
        species[31] =
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
        /*species 3: CH3O */
        species[3] =
            +9.786011000000000e+02 * invT
            -1.104597600000000e+01
            -2.106204000000000e+00 * tc[0]
            -3.608297500000000e-03 * tc[1]
            -8.897453333333333e-07 * tc[2]
            +6.148030000000000e-10 * tc[3]
            -1.037805500000000e-13 * tc[4];
        /*species 4: C2H5 */
        species[4] =
            +1.287040000000000e+04 * invT
            -9.447498000000000e+00
            -2.690702000000000e+00 * tc[0]
            -4.359566500000000e-03 * tc[1]
            -7.366398333333332e-07 * tc[2]
            -7.782252500000001e-11 * tc[3]
            +1.963886500000000e-13 * tc[4];
        /*species 6: HCCO */
        species[6] =
            +1.965892000000000e+04 * invT
            +4.566121099999999e+00
            -5.047965000000000e+00 * tc[0]
            -2.226739000000000e-03 * tc[1]
            -3.780471666666667e-08 * tc[2]
            +1.235079166666667e-10 * tc[3]
            -1.125371000000000e-14 * tc[4];
        /*species 7: C2H3 */
        species[7] =
            +3.335225000000000e+04 * invT
            -9.096924000000001e+00
            -2.459276000000000e+00 * tc[0]
            -3.685738000000000e-03 * tc[1]
            -3.516455000000000e-07 * tc[2]
            +1.101368333333333e-10 * tc[3]
            +5.923920000000000e-14 * tc[4];
        /*species 8: CH2CHO */
        species[8] =
            +1.521477000000000e+03 * invT
            -6.149227999999999e+00
            -3.409062000000000e+00 * tc[0]
            -5.369285000000000e-03 * tc[1]
            -3.152486666666667e-07 * tc[2]
            +5.965485833333333e-10 * tc[3]
            -1.433692500000000e-13 * tc[4];
        /*species 11: C3H2 */
        species[11] =
            +6.350421000000000e+04 * invT
            -5.702732000000000e+00
            -3.166714000000000e+00 * tc[0]
            -1.241286000000000e-02 * tc[1]
            +7.652728333333333e-06 * tc[2]
            -3.556682500000000e-09 * tc[3]
            +7.410759999999999e-13 * tc[4];
        /*species 12: C3H3 */
        species[12] =
            +3.988883000000000e+04 * invT
            +4.168745100000000e+00
            -4.754200000000000e+00 * tc[0]
            -5.540140000000000e-03 * tc[1]
            -4.655538333333333e-08 * tc[2]
            +4.566010000000000e-10 * tc[3]
            -9.748145000000000e-14 * tc[4];
        /*species 13: NXC3H7 */
        species[13] =
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
        /*species 3: CH3O */
        species[3] =
            +1.278325000000000e+02 * invT
            +8.412250000000001e-01
            -3.770800000000000e+00 * tc[0]
            -3.935748500000000e-03 * tc[1]
            +4.427306666666667e-07 * tc[2]
            -3.287025833333333e-11 * tc[3]
            +1.056308000000000e-15 * tc[4];
        /*species 4: C2H5 */
        species[4] =
            +1.067455000000000e+04 * invT
            +2.197137000000000e+01
            -7.190480000000000e+00 * tc[0]
            -3.242038500000000e-03 * tc[1]
            +1.071344166666667e-07 * tc[2]
            +1.956565833333333e-11 * tc[3]
            -1.940438500000000e-15 * tc[4];
        /*species 6: HCCO */
        species[6] =
            +1.901513000000000e+04 * invT
            +1.582933500000000e+01
            -6.758073000000000e+00 * tc[0]
            -1.000200000000000e-03 * tc[1]
            +3.379345000000000e-08 * tc[2]
            +8.676100000000000e-12 * tc[3]
            -9.825825000000000e-16 * tc[4];
        /*species 7: C2H3 */
        species[7] =
            +3.185435000000000e+04 * invT
            +1.446378100000000e+01
            -5.933468000000000e+00 * tc[0]
            -2.008873000000000e-03 * tc[1]
            +6.611233333333333e-08 * tc[2]
            +1.201055833333333e-11 * tc[3]
            -1.189322000000000e-15 * tc[4];
        /*species 8: CH2CHO */
        species[8] =
            +4.903218000000000e+02 * invT
            +1.102092100000000e+01
            -5.975670000000000e+00 * tc[0]
            -4.065295500000000e-03 * tc[1]
            +4.572706666666667e-07 * tc[2]
            -3.391920000000000e-11 * tc[3]
            +1.088008500000000e-15 * tc[4];
        /*species 11: C3H2 */
        species[11] =
            +6.259722000000000e+04 * invT
            +2.003988100000000e+01
            -7.670981000000000e+00 * tc[0]
            -1.374374500000000e-03 * tc[1]
            +7.284905000000000e-08 * tc[2]
            +5.379665833333334e-12 * tc[3]
            -8.319435000000000e-16 * tc[4];
        /*species 12: C3H3 */
        species[12] =
            +3.847420000000000e+04 * invT
            +3.061023700000000e+01
            -8.831047000000000e+00 * tc[0]
            -2.178597500000000e-03 * tc[1]
            +6.848445000000000e-08 * tc[2]
            +1.973935833333333e-11 * tc[3]
            -2.188260000000000e-15 * tc[4];
        /*species 13: NXC3H7 */
        species[13] =
            +7.579402000000000e+03 * invT
            +2.733440100000000e+01
            -7.978291000000000e+00 * tc[0]
            -7.880565000000001e-03 * tc[1]
            +8.622071666666666e-07 * tc[2]
            -6.203243333333333e-11 * tc[3]
            +1.912489000000000e-15 * tc[4];
    }

    /*species with midpoint at T=1390 kelvin */
    if (T < 1390) {
        /*species 5: CH3O2H */
        species[5] =
            -1.771979260000000e+04 * invT
            -6.021811320000000e+00
            -3.234428170000000e+00 * tc[0]
            -9.506488350000000e-03 * tc[1]
            +1.889771450000000e-06 * tc[2]
            -2.835888775000000e-10 * tc[3]
            +2.059151110000000e-14 * tc[4];
    } else {
        /*species 5: CH3O2H */
        species[5] =
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
        /*species 9: C2H5O */
        species[9] =
            -3.352529250000000e+03 * invT
            -2.231351709200000e+01
            -4.944207080000000e-01 * tc[0]
            -1.358872170000000e-02 * tc[1]
            +2.765150166666667e-06 * tc[2]
            -4.293368333333333e-10 * tc[3]
            +3.242484575000000e-14 * tc[4];
    } else {
        /*species 9: C2H5O */
        species[9] =
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
        /*species 10: C2H5O2 */
        species[10] =
            -5.038807580000000e+03 * invT
            -1.420893532000000e+01
            -2.268461880000000e+00 * tc[0]
            -1.384712890000000e-02 * tc[1]
            +2.846735100000000e-06 * tc[2]
            -4.899598983333333e-10 * tc[3]
            +4.604495345000000e-14 * tc[4];
    } else {
        /*species 10: C2H5O2 */
        species[10] =
            -7.824817950000000e+03 * invT
            +3.254826223000000e+01
            -9.486960229999999e+00 * tc[0]
            -6.223627250000000e-03 * tc[1]
            +7.202692933333333e-07 * tc[2]
            -5.646525275000000e-11 * tc[3]
            +1.978922840000000e-15 * tc[4];
    }

    /*species with midpoint at T=1395 kelvin */
    if (T < 1395) {
        /*species 14: PXC4H9 */
        species[14] =
            +7.689452480000000e+03 * invT
            -2.912305292500000e+01
            +4.377797250000000e-01 * tc[0]
            -2.394861820000000e-02 * tc[1]
            +5.233719316666667e-06 * tc[2]
            -9.148872666666667e-10 * tc[3]
            +8.100533200000001e-14 * tc[4];
    } else {
        /*species 14: PXC4H9 */
        species[14] =
            +3.172319420000000e+03 * invT
            +5.149359040000000e+01
            -1.215100820000000e+01 * tc[0]
            -9.715535850000000e-03 * tc[1]
            +1.102629916666667e-06 * tc[2]
            -8.531261333333333e-11 * tc[3]
            +2.962648535000000e-15 * tc[4];
    }

    /*species with midpoint at T=1385 kelvin */
    if (T < 1385) {
        /*species 15: PXC4H9O2 */
        species[15] =
            -1.083581030000000e+04 * invT
            -1.940667840000000e+01
            -1.943636500000000e+00 * tc[0]
            -2.577565815000000e-02 * tc[1]
            +5.471406666666667e-06 * tc[2]
            -9.422071666666667e-10 * tc[3]
            +8.505930300000000e-14 * tc[4];
    } else {
        /*species 15: PXC4H9O2 */
        species[15] =
            -1.601460540000000e+04 * invT
            +6.982339730000000e+01
            -1.578454480000000e+01 * tc[0]
            -1.076054550000000e-02 * tc[1]
            +1.241515028333333e-06 * tc[2]
            -9.713172583333334e-11 * tc[3]
            +3.399428045000000e-15 * tc[4];
    }

    /*species with midpoint at T=1392 kelvin */
    if (T < 1392) {
        /*species 16: C5H9 */
        species[16] =
            +1.255898240000000e+04 * invT
            -3.402426990000000e+01
            +1.380139500000000e+00 * tc[0]
            -2.788042435000000e-02 * tc[1]
            +6.169065466666666e-06 * tc[2]
            -1.057365841666667e-09 * tc[3]
            +8.926941750000000e-14 * tc[4];
        /*species 17: C5H10X1 */
        species[17] =
            -4.465466660000000e+03 * invT
            -3.333621381000000e+01
            +1.062234810000000e+00 * tc[0]
            -2.871091470000000e-02 * tc[1]
            +6.241448166666667e-06 * tc[2]
            -1.061374908333333e-09 * tc[3]
            +8.980489449999999e-14 * tc[4];
        /*species 19: C6H12X1 */
        species[19] =
            -7.343686170000000e+03 * invT
            -3.666482115000000e+01
            +1.352752050000000e+00 * tc[0]
            -3.493277130000000e-02 * tc[1]
            +7.656800366666667e-06 * tc[2]
            -1.308061191666667e-09 * tc[3]
            +1.106480875000000e-13 * tc[4];
    } else {
        /*species 16: C5H9 */
        species[16] =
            +7.004961350000000e+03 * invT
            +6.563622270000000e+01
            -1.418604540000000e+01 * tc[0]
            -1.035644495000000e-02 * tc[1]
            +1.178267695000000e-06 * tc[2]
            -9.133927750000000e-11 * tc[3]
            +3.176611040000000e-15 * tc[4];
        /*species 17: C5H10X1 */
        species[17] =
            -1.008982050000000e+04 * invT
            +6.695354750000000e+01
            -1.458515390000000e+01 * tc[0]
            -1.120362355000000e-02 * tc[1]
            +1.272246708333333e-06 * tc[2]
            -9.849080500000001e-11 * tc[3]
            +3.421925695000000e-15 * tc[4];
        /*species 19: C6H12X1 */
        species[19] =
            -1.420628600000000e+04 * invT
            +8.621563800000001e+01
            -1.783375290000000e+01 * tc[0]
            -1.336888290000000e-02 * tc[1]
            +1.516727955000000e-06 * tc[2]
            -1.173498066666667e-10 * tc[3]
            +4.075621220000000e-15 * tc[4];
    }

    /*species with midpoint at T=1397 kelvin */
    if (T < 1397) {
        /*species 18: C5H11X1 */
        species[18] =
            +4.839953030000000e+03 * invT
            -3.346275221200000e+01
            +9.052559120000000e-01 * tc[0]
            -3.053164260000000e-02 * tc[1]
            +6.824863750000000e-06 * tc[2]
            -1.217445583333333e-09 * tc[3]
            +1.094298075000000e-13 * tc[4];
    } else {
        /*species 18: C5H11X1 */
        species[18] =
            -9.232416370000000e+02 * invT
            +7.027635990000000e+01
            -1.532347400000000e+01 * tc[0]
            -1.195206000000000e-02 * tc[1]
            +1.357952698333333e-06 * tc[2]
            -1.051468633333333e-10 * tc[3]
            +3.653386675000001e-15 * tc[4];
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
        /*species 12: CH2GSG */
        species[12] =
            +4.98936800e+04 * invT
            +2.91373293e+00
            -3.97126500e+00 * tc[0]
            +8.49544500e-05 * tc[1]
            -1.70894833e-07 * tc[2]
            -2.07712583e-10 * tc[3]
            +9.90633000e-14 * tc[4];
        /*species 13: CH3 */
        species[13] =
            +1.64237800e+04 * invT
            -5.35935100e+00
            -2.43044300e+00 * tc[0]
            -5.56205000e-03 * tc[1]
            +2.80036667e-06 * tc[2]
            -1.35152417e-09 * tc[3]
            +2.93247650e-13 * tc[4];
        /*species 14: CH4 */
        species[14] =
            -9.82522900e+03 * invT
            -1.39434485e+01
            -7.78741500e-01 * tc[0]
            -8.73834000e-03 * tc[1]
            +4.63901500e-06 * tc[2]
            -2.54142333e-09 * tc[3]
            +6.11965500e-13 * tc[4];
        /*species 15: CH3OH */
        species[15] =
            -2.53534800e+04 * invT
            -9.57251500e+00
            -2.66011500e+00 * tc[0]
            -3.67075400e-03 * tc[1]
            -1.19500850e-06 * tc[2]
            +7.32766167e-10 * tc[3]
            -1.19528500e-13 * tc[4];
        /*species 17: CH2CO */
        species[17] =
            -7.63263700e+03 * invT
            -6.69858200e+00
            -2.97497100e+00 * tc[0]
            -6.05935500e-03 * tc[1]
            +3.90841000e-07 * tc[2]
            +5.38890417e-10 * tc[3]
            -1.95282450e-13 * tc[4];
        /*species 20: C2H2 */
        species[20] =
            +2.61244400e+04 * invT
            -7.79181600e+00
            -2.01356200e+00 * tc[0]
            -7.59522500e-03 * tc[1]
            +2.69386500e-06 * tc[2]
            -7.56582667e-10 * tc[3]
            +9.56373000e-14 * tc[4];
        /*species 22: C2H4 */
        species[22] =
            +5.57304600e+03 * invT
            -2.60729780e+01
            +8.61488000e-01 * tc[0]
            -1.39808150e-02 * tc[1]
            +5.64779500e-06 * tc[2]
            -2.32096000e-09 * tc[3]
            +4.86893950e-13 * tc[4];
        /*species 23: CH3CO */
        species[23] =
            -4.10850800e+03 * invT
            -9.10357200e+00
            -3.12527800e+00 * tc[0]
            -4.88911000e-03 * tc[1]
            -7.53574667e-07 * tc[2]
            +7.50788500e-10 * tc[3]
            -1.59685900e-13 * tc[4];
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
        /*species 12: CH2GSG */
        species[12] =
            +4.98497500e+04 * invT
            +8.66319000e-01
            -3.55288900e+00 * tc[0]
            -1.03339400e-03 * tc[1]
            +3.19019333e-08 * tc[2]
            +9.20560833e-12 * tc[3]
            -1.01067500e-15 * tc[4];
        /*species 13: CH3 */
        species[13] =
            +1.64378100e+04 * invT
            -3.60864500e+00
            -2.84405200e+00 * tc[0]
            -3.06898700e-03 * tc[1]
            +3.71724167e-07 * tc[2]
            -3.15430083e-11 * tc[3]
            +1.22607950e-15 * tc[4];
        /*species 14: CH4 */
        species[14] =
            -1.00807900e+04 * invT
            -8.93991600e+00
            -1.68347900e+00 * tc[0]
            -5.11862000e-03 * tc[1]
            +6.45854833e-07 * tc[2]
            -5.65465417e-11 * tc[3]
            +2.25171150e-15 * tc[4];
        /*species 15: CH3OH */
        species[15] =
            -2.61579100e+04 * invT
            +6.50865000e-01
            -4.02906100e+00 * tc[0]
            -4.68829650e-03 * tc[1]
            +5.08375667e-07 * tc[2]
            -3.63232750e-11 * tc[3]
            +1.11236150e-15 * tc[4];
        /*species 17: CH2CO */
        species[17] =
            -8.58340200e+03 * invT
            +1.26963980e+01
            -6.03881700e+00 * tc[0]
            -2.90242000e-03 * tc[1]
            +3.20159000e-07 * tc[2]
            -2.32873750e-11 * tc[3]
            +7.29434000e-16 * tc[4];
        /*species 20: C2H2 */
        species[20] =
            +2.56676600e+04 * invT
            +6.23710800e+00
            -4.43677000e+00 * tc[0]
            -2.68801950e-03 * tc[1]
            +3.18802833e-07 * tc[2]
            -2.73864917e-11 * tc[3]
            +1.07835500e-15 * tc[4];
        /*species 22: C2H4 */
        species[22] =
            +4.42828900e+03 * invT
            +2.98030000e-01
            -3.52841900e+00 * tc[0]
            -5.74259000e-03 * tc[1]
            +7.36397500e-07 * tc[2]
            -6.53716750e-11 * tc[3]
            +2.63342400e-15 * tc[4];
        /*species 23: CH3CO */
        species[23] =
            -5.18786300e+03 * invT
            +7.88722800e+00
            -5.61227900e+00 * tc[0]
            -4.22494300e-03 * tc[1]
            +4.75691167e-07 * tc[2]
            -3.53198000e-11 * tc[3]
            +1.13420200e-15 * tc[4];
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
        /*species 16: C2H6 */
        species[16] =
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
        /*species 16: C2H6 */
        species[16] =
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
        /*species 18: HOCHO */
        species[18] =
            -4.64616504e+04 * invT
            -1.68530979e+01
            -1.43548185e+00 * tc[0]
            -8.16815080e-03 * tc[1]
            +1.77095702e-06 * tc[2]
            -2.76777481e-10 * tc[3]
            +2.01088051e-14 * tc[4];
    } else {
        /*species 18: HOCHO */
        species[18] =
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
        /*species 19: CH3O2 */
        species[19] =
            -6.84394259e+02 * invT
            -1.90183414e+00
            -4.26146906e+00 * tc[0]
            -5.04367995e-03 * tc[1]
            +5.35843640e-07 * tc[2]
            -1.74507723e-11 * tc[3]
            -2.09169552e-15 * tc[4];
    } else {
        /*species 19: CH3O2 */
        species[19] =
            -1.53574838e+03 * invT
            +9.67751777e+00
            -5.95787891e+00 * tc[0]
            -3.95364313e-03 * tc[1]
            +4.47077057e-07 * tc[2]
            -3.44909447e-11 * tc[3]
            +1.19503665e-15 * tc[4];
    }

    /*species with midpoint at T=1388 kelvin */
    if (T < 1388) {
        /*species 21: C3H6 */
        species[21] =
            +1.06688164e+03 * invT
            -2.25057582e+01
            -3.94615444e-01 * tc[0]
            -1.44553831e-02 * tc[1]
            +2.58144680e-06 * tc[2]
            -3.24011841e-10 * tc[3]
            +1.68945176e-14 * tc[4];
    } else {
        /*species 21: C3H6 */
        species[21] =
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
        /*species 24: C3H4XA */
        species[24] =
            +2.25124300e+04 * invT
            -8.39587100e+00
            -2.53983100e+00 * tc[0]
            -8.16718500e-03 * tc[1]
            +2.94158333e-07 * tc[2]
            +3.87280417e-10 * tc[3]
            -8.64565500e-14 * tc[4];
    } else {
        /*species 24: C3H4XA */
        species[24] =
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
        /*species 25: C3H5XA */
        species[25] =
            +1.93834226e+04 * invT
            -2.68358451e+01
            +5.29131958e-01 * tc[0]
            -1.67279550e-02 * tc[1]
            +4.22335045e-06 * tc[2]
            -8.57214617e-10 * tc[3]
            +8.66291700e-14 * tc[4];
    } else {
        /*species 25: C3H5XA */
        species[25] =
            +1.63576092e+04 * invT
            +3.00397846e+01
            -8.45883958e+00 * tc[0]
            -5.63477415e-03 * tc[1]
            +6.39654773e-07 * tc[2]
            -4.95049266e-11 * tc[3]
            +1.71959015e-15 * tc[4];
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
    }

    /*species with midpoint at T=1382 kelvin */
    if (T < 1382) {
        /*species 30: C7H15X2 */
        species[30] =
            -2.35605303e+03 * invT
            -3.47700662e+01
            +3.79155767e-02 * tc[0]
            -3.78363285e-02 * tc[1]
            +6.79122723e-06 * tc[2]
            -7.77232453e-10 * tc[3]
            +2.46180373e-14 * tc[4];
    } else {
        /*species 30: C7H15X2 */
        species[30] =
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
        /*species 31: NXC7H16 */
        species[31] =
            -2.56586565e+04 * invT
            -3.76416531e+01
            +1.26836187e+00 * tc[0]
            -4.27177910e-02 * tc[1]
            +8.75577977e-06 * tc[2]
            -1.35788101e-09 * tc[3]
            +1.01197462e-13 * tc[4];
    } else {
        /*species 31: NXC7H16 */
        species[31] =
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
        /*species 12: CH2GSG */
        species[12] =
            +2.97126500e+00
            -1.69908900e-04 * tc[1]
            +1.02536900e-06 * tc[2]
            +2.49255100e-09 * tc[3]
            -1.98126600e-12 * tc[4];
        /*species 13: CH3 */
        species[13] =
            +1.43044300e+00
            +1.11241000e-02 * tc[1]
            -1.68022000e-05 * tc[2]
            +1.62182900e-08 * tc[3]
            -5.86495300e-12 * tc[4];
        /*species 14: CH4 */
        species[14] =
            -2.21258500e-01
            +1.74766800e-02 * tc[1]
            -2.78340900e-05 * tc[2]
            +3.04970800e-08 * tc[3]
            -1.22393100e-11 * tc[4];
        /*species 15: CH3OH */
        species[15] =
            +1.66011500e+00
            +7.34150800e-03 * tc[1]
            +7.17005100e-06 * tc[2]
            -8.79319400e-09 * tc[3]
            +2.39057000e-12 * tc[4];
        /*species 17: CH2CO */
        species[17] =
            +1.97497100e+00
            +1.21187100e-02 * tc[1]
            -2.34504600e-06 * tc[2]
            -6.46668500e-09 * tc[3]
            +3.90564900e-12 * tc[4];
        /*species 20: C2H2 */
        species[20] =
            +1.01356200e+00
            +1.51904500e-02 * tc[1]
            -1.61631900e-05 * tc[2]
            +9.07899200e-09 * tc[3]
            -1.91274600e-12 * tc[4];
        /*species 22: C2H4 */
        species[22] =
            -1.86148800e+00
            +2.79616300e-02 * tc[1]
            -3.38867700e-05 * tc[2]
            +2.78515200e-08 * tc[3]
            -9.73787900e-12 * tc[4];
        /*species 23: CH3CO */
        species[23] =
            +2.12527800e+00
            +9.77822000e-03 * tc[1]
            +4.52144800e-06 * tc[2]
            -9.00946200e-09 * tc[3]
            +3.19371800e-12 * tc[4];
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
        /*species 12: CH2GSG */
        species[12] =
            +2.55288900e+00
            +2.06678800e-03 * tc[1]
            -1.91411600e-07 * tc[2]
            -1.10467300e-10 * tc[3]
            +2.02135000e-14 * tc[4];
        /*species 13: CH3 */
        species[13] =
            +1.84405200e+00
            +6.13797400e-03 * tc[1]
            -2.23034500e-06 * tc[2]
            +3.78516100e-10 * tc[3]
            -2.45215900e-14 * tc[4];
        /*species 14: CH4 */
        species[14] =
            +6.83479000e-01
            +1.02372400e-02 * tc[1]
            -3.87512900e-06 * tc[2]
            +6.78558500e-10 * tc[3]
            -4.50342300e-14 * tc[4];
        /*species 15: CH3OH */
        species[15] =
            +3.02906100e+00
            +9.37659300e-03 * tc[1]
            -3.05025400e-06 * tc[2]
            +4.35879300e-10 * tc[3]
            -2.22472300e-14 * tc[4];
        /*species 17: CH2CO */
        species[17] =
            +5.03881700e+00
            +5.80484000e-03 * tc[1]
            -1.92095400e-06 * tc[2]
            +2.79448500e-10 * tc[3]
            -1.45886800e-14 * tc[4];
        /*species 20: C2H2 */
        species[20] =
            +3.43677000e+00
            +5.37603900e-03 * tc[1]
            -1.91281700e-06 * tc[2]
            +3.28637900e-10 * tc[3]
            -2.15671000e-14 * tc[4];
        /*species 22: C2H4 */
        species[22] =
            +2.52841900e+00
            +1.14851800e-02 * tc[1]
            -4.41838500e-06 * tc[2]
            +7.84460100e-10 * tc[3]
            -5.26684800e-14 * tc[4];
        /*species 23: CH3CO */
        species[23] =
            +4.61227900e+00
            +8.44988600e-03 * tc[1]
            -2.85414700e-06 * tc[2]
            +4.23837600e-10 * tc[3]
            -2.26840400e-14 * tc[4];
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
        /*species 16: C2H6 */
        species[16] =
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
        /*species 16: C2H6 */
        species[16] =
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
        /*species 18: HOCHO */
        species[18] =
            +4.35481850e-01
            +1.63363016e-02 * tc[1]
            -1.06257421e-05 * tc[2]
            +3.32132977e-09 * tc[3]
            -4.02176103e-13 * tc[4];
    } else {
        /*species 18: HOCHO */
        species[18] =
            +5.68733013e+00
            +5.14289368e-03 * tc[1]
            -1.82238513e-06 * tc[2]
            +2.89719163e-10 * tc[3]
            -1.70892199e-14 * tc[4];
    }

    /*species with midpoint at T=1385 kelvin */
    if (T < 1385) {
        /*species 19: CH3O2 */
        species[19] =
            +3.26146906e+00
            +1.00873599e-02 * tc[1]
            -3.21506184e-06 * tc[2]
            +2.09409267e-10 * tc[3]
            +4.18339103e-14 * tc[4];
    } else {
        /*species 19: CH3O2 */
        species[19] =
            +4.95787891e+00
            +7.90728626e-03 * tc[1]
            -2.68246234e-06 * tc[2]
            +4.13891337e-10 * tc[3]
            -2.39007330e-14 * tc[4];
    }

    /*species with midpoint at T=1388 kelvin */
    if (T < 1388) {
        /*species 21: C3H6 */
        species[21] =
            -6.05384556e-01
            +2.89107662e-02 * tc[1]
            -1.54886808e-05 * tc[2]
            +3.88814209e-09 * tc[3]
            -3.37890352e-13 * tc[4];
    } else {
        /*species 21: C3H6 */
        species[21] =
            +7.01595958e+00
            +1.37023634e-02 * tc[1]
            -4.66249733e-06 * tc[2]
            +7.21254402e-10 * tc[3]
            -4.17370126e-14 * tc[4];
    }

    /*species with midpoint at T=1400 kelvin */
    if (T < 1400) {
        /*species 24: C3H4XA */
        species[24] =
            +1.53983100e+00
            +1.63343700e-02 * tc[1]
            -1.76495000e-06 * tc[2]
            -4.64736500e-09 * tc[3]
            +1.72913100e-12 * tc[4];
    } else {
        /*species 24: C3H4XA */
        species[24] =
            +8.77625600e+00
            +5.30213800e-03 * tc[1]
            -3.70111800e-07 * tc[2]
            -3.02638600e-10 * tc[3]
            +5.08958100e-14 * tc[4];
    }

    /*species with midpoint at T=1397 kelvin */
    if (T < 1397) {
        /*species 25: C3H5XA */
        species[25] =
            -1.52913196e+00
            +3.34559100e-02 * tc[1]
            -2.53401027e-05 * tc[2]
            +1.02865754e-08 * tc[3]
            -1.73258340e-12 * tc[4];
    } else {
        /*species 25: C3H5XA */
        species[25] =
            +7.45883958e+00
            +1.12695483e-02 * tc[1]
            -3.83792864e-06 * tc[2]
            +5.94059119e-10 * tc[3]
            -3.43918030e-14 * tc[4];
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
    }

    /*species with midpoint at T=1382 kelvin */
    if (T < 1382) {
        /*species 30: C7H15X2 */
        species[30] =
            -1.03791558e+00
            +7.56726570e-02 * tc[1]
            -4.07473634e-05 * tc[2]
            +9.32678943e-09 * tc[3]
            -4.92360745e-13 * tc[4];
    } else {
        /*species 30: C7H15X2 */
        species[30] =
            +2.06368842e+01
            +3.23324804e-02 * tc[1]
            -1.09273807e-05 * tc[2]
            +1.68357060e-09 * tc[3]
            -9.71774091e-14 * tc[4];
    }

    /*species with midpoint at T=1391 kelvin */
    if (T < 1391) {
        /*species 31: NXC7H16 */
        species[31] =
            -2.26836187e+00
            +8.54355820e-02 * tc[1]
            -5.25346786e-05 * tc[2]
            +1.62945721e-08 * tc[3]
            -2.02394925e-12 * tc[4];
    } else {
        /*species 31: NXC7H16 */
        species[31] =
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
        /*species 12: CH2GSG */
        species[12] =
            +3.97126500e+00
            -1.69908900e-04 * tc[1]
            +1.02536900e-06 * tc[2]
            +2.49255100e-09 * tc[3]
            -1.98126600e-12 * tc[4];
        /*species 13: CH3 */
        species[13] =
            +2.43044300e+00
            +1.11241000e-02 * tc[1]
            -1.68022000e-05 * tc[2]
            +1.62182900e-08 * tc[3]
            -5.86495300e-12 * tc[4];
        /*species 14: CH4 */
        species[14] =
            +7.78741500e-01
            +1.74766800e-02 * tc[1]
            -2.78340900e-05 * tc[2]
            +3.04970800e-08 * tc[3]
            -1.22393100e-11 * tc[4];
        /*species 15: CH3OH */
        species[15] =
            +2.66011500e+00
            +7.34150800e-03 * tc[1]
            +7.17005100e-06 * tc[2]
            -8.79319400e-09 * tc[3]
            +2.39057000e-12 * tc[4];
        /*species 17: CH2CO */
        species[17] =
            +2.97497100e+00
            +1.21187100e-02 * tc[1]
            -2.34504600e-06 * tc[2]
            -6.46668500e-09 * tc[3]
            +3.90564900e-12 * tc[4];
        /*species 20: C2H2 */
        species[20] =
            +2.01356200e+00
            +1.51904500e-02 * tc[1]
            -1.61631900e-05 * tc[2]
            +9.07899200e-09 * tc[3]
            -1.91274600e-12 * tc[4];
        /*species 22: C2H4 */
        species[22] =
            -8.61488000e-01
            +2.79616300e-02 * tc[1]
            -3.38867700e-05 * tc[2]
            +2.78515200e-08 * tc[3]
            -9.73787900e-12 * tc[4];
        /*species 23: CH3CO */
        species[23] =
            +3.12527800e+00
            +9.77822000e-03 * tc[1]
            +4.52144800e-06 * tc[2]
            -9.00946200e-09 * tc[3]
            +3.19371800e-12 * tc[4];
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
        /*species 12: CH2GSG */
        species[12] =
            +3.55288900e+00
            +2.06678800e-03 * tc[1]
            -1.91411600e-07 * tc[2]
            -1.10467300e-10 * tc[3]
            +2.02135000e-14 * tc[4];
        /*species 13: CH3 */
        species[13] =
            +2.84405200e+00
            +6.13797400e-03 * tc[1]
            -2.23034500e-06 * tc[2]
            +3.78516100e-10 * tc[3]
            -2.45215900e-14 * tc[4];
        /*species 14: CH4 */
        species[14] =
            +1.68347900e+00
            +1.02372400e-02 * tc[1]
            -3.87512900e-06 * tc[2]
            +6.78558500e-10 * tc[3]
            -4.50342300e-14 * tc[4];
        /*species 15: CH3OH */
        species[15] =
            +4.02906100e+00
            +9.37659300e-03 * tc[1]
            -3.05025400e-06 * tc[2]
            +4.35879300e-10 * tc[3]
            -2.22472300e-14 * tc[4];
        /*species 17: CH2CO */
        species[17] =
            +6.03881700e+00
            +5.80484000e-03 * tc[1]
            -1.92095400e-06 * tc[2]
            +2.79448500e-10 * tc[3]
            -1.45886800e-14 * tc[4];
        /*species 20: C2H2 */
        species[20] =
            +4.43677000e+00
            +5.37603900e-03 * tc[1]
            -1.91281700e-06 * tc[2]
            +3.28637900e-10 * tc[3]
            -2.15671000e-14 * tc[4];
        /*species 22: C2H4 */
        species[22] =
            +3.52841900e+00
            +1.14851800e-02 * tc[1]
            -4.41838500e-06 * tc[2]
            +7.84460100e-10 * tc[3]
            -5.26684800e-14 * tc[4];
        /*species 23: CH3CO */
        species[23] =
            +5.61227900e+00
            +8.44988600e-03 * tc[1]
            -2.85414700e-06 * tc[2]
            +4.23837600e-10 * tc[3]
            -2.26840400e-14 * tc[4];
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
        /*species 16: C2H6 */
        species[16] =
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
        /*species 16: C2H6 */
        species[16] =
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
        /*species 18: HOCHO */
        species[18] =
            +1.43548185e+00
            +1.63363016e-02 * tc[1]
            -1.06257421e-05 * tc[2]
            +3.32132977e-09 * tc[3]
            -4.02176103e-13 * tc[4];
    } else {
        /*species 18: HOCHO */
        species[18] =
            +6.68733013e+00
            +5.14289368e-03 * tc[1]
            -1.82238513e-06 * tc[2]
            +2.89719163e-10 * tc[3]
            -1.70892199e-14 * tc[4];
    }

    /*species with midpoint at T=1385 kelvin */
    if (T < 1385) {
        /*species 19: CH3O2 */
        species[19] =
            +4.26146906e+00
            +1.00873599e-02 * tc[1]
            -3.21506184e-06 * tc[2]
            +2.09409267e-10 * tc[3]
            +4.18339103e-14 * tc[4];
    } else {
        /*species 19: CH3O2 */
        species[19] =
            +5.95787891e+00
            +7.90728626e-03 * tc[1]
            -2.68246234e-06 * tc[2]
            +4.13891337e-10 * tc[3]
            -2.39007330e-14 * tc[4];
    }

    /*species with midpoint at T=1388 kelvin */
    if (T < 1388) {
        /*species 21: C3H6 */
        species[21] =
            +3.94615444e-01
            +2.89107662e-02 * tc[1]
            -1.54886808e-05 * tc[2]
            +3.88814209e-09 * tc[3]
            -3.37890352e-13 * tc[4];
    } else {
        /*species 21: C3H6 */
        species[21] =
            +8.01595958e+00
            +1.37023634e-02 * tc[1]
            -4.66249733e-06 * tc[2]
            +7.21254402e-10 * tc[3]
            -4.17370126e-14 * tc[4];
    }

    /*species with midpoint at T=1400 kelvin */
    if (T < 1400) {
        /*species 24: C3H4XA */
        species[24] =
            +2.53983100e+00
            +1.63343700e-02 * tc[1]
            -1.76495000e-06 * tc[2]
            -4.64736500e-09 * tc[3]
            +1.72913100e-12 * tc[4];
    } else {
        /*species 24: C3H4XA */
        species[24] =
            +9.77625600e+00
            +5.30213800e-03 * tc[1]
            -3.70111800e-07 * tc[2]
            -3.02638600e-10 * tc[3]
            +5.08958100e-14 * tc[4];
    }

    /*species with midpoint at T=1397 kelvin */
    if (T < 1397) {
        /*species 25: C3H5XA */
        species[25] =
            -5.29131958e-01
            +3.34559100e-02 * tc[1]
            -2.53401027e-05 * tc[2]
            +1.02865754e-08 * tc[3]
            -1.73258340e-12 * tc[4];
    } else {
        /*species 25: C3H5XA */
        species[25] =
            +8.45883958e+00
            +1.12695483e-02 * tc[1]
            -3.83792864e-06 * tc[2]
            +5.94059119e-10 * tc[3]
            -3.43918030e-14 * tc[4];
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
    }

    /*species with midpoint at T=1382 kelvin */
    if (T < 1382) {
        /*species 30: C7H15X2 */
        species[30] =
            -3.79155767e-02
            +7.56726570e-02 * tc[1]
            -4.07473634e-05 * tc[2]
            +9.32678943e-09 * tc[3]
            -4.92360745e-13 * tc[4];
    } else {
        /*species 30: C7H15X2 */
        species[30] =
            +2.16368842e+01
            +3.23324804e-02 * tc[1]
            -1.09273807e-05 * tc[2]
            +1.68357060e-09 * tc[3]
            -9.71774091e-14 * tc[4];
    }

    /*species with midpoint at T=1391 kelvin */
    if (T < 1391) {
        /*species 31: NXC7H16 */
        species[31] =
            -1.26836187e+00
            +8.54355820e-02 * tc[1]
            -5.25346786e-05 * tc[2]
            +1.62945721e-08 * tc[3]
            -2.02394925e-12 * tc[4];
    } else {
        /*species 31: NXC7H16 */
        species[31] =
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
        /*species 12: CH2GSG */
        species[12] =
            +2.97126500e+00
            -8.49544500e-05 * tc[1]
            +3.41789667e-07 * tc[2]
            +6.23137750e-10 * tc[3]
            -3.96253200e-13 * tc[4]
            +4.98936800e+04 * invT;
        /*species 13: CH3 */
        species[13] =
            +1.43044300e+00
            +5.56205000e-03 * tc[1]
            -5.60073333e-06 * tc[2]
            +4.05457250e-09 * tc[3]
            -1.17299060e-12 * tc[4]
            +1.64237800e+04 * invT;
        /*species 14: CH4 */
        species[14] =
            -2.21258500e-01
            +8.73834000e-03 * tc[1]
            -9.27803000e-06 * tc[2]
            +7.62427000e-09 * tc[3]
            -2.44786200e-12 * tc[4]
            -9.82522900e+03 * invT;
        /*species 15: CH3OH */
        species[15] =
            +1.66011500e+00
            +3.67075400e-03 * tc[1]
            +2.39001700e-06 * tc[2]
            -2.19829850e-09 * tc[3]
            +4.78114000e-13 * tc[4]
            -2.53534800e+04 * invT;
        /*species 17: CH2CO */
        species[17] =
            +1.97497100e+00
            +6.05935500e-03 * tc[1]
            -7.81682000e-07 * tc[2]
            -1.61667125e-09 * tc[3]
            +7.81129800e-13 * tc[4]
            -7.63263700e+03 * invT;
        /*species 20: C2H2 */
        species[20] =
            +1.01356200e+00
            +7.59522500e-03 * tc[1]
            -5.38773000e-06 * tc[2]
            +2.26974800e-09 * tc[3]
            -3.82549200e-13 * tc[4]
            +2.61244400e+04 * invT;
        /*species 22: C2H4 */
        species[22] =
            -1.86148800e+00
            +1.39808150e-02 * tc[1]
            -1.12955900e-05 * tc[2]
            +6.96288000e-09 * tc[3]
            -1.94757580e-12 * tc[4]
            +5.57304600e+03 * invT;
        /*species 23: CH3CO */
        species[23] =
            +2.12527800e+00
            +4.88911000e-03 * tc[1]
            +1.50714933e-06 * tc[2]
            -2.25236550e-09 * tc[3]
            +6.38743600e-13 * tc[4]
            -4.10850800e+03 * invT;
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
        /*species 12: CH2GSG */
        species[12] =
            +2.55288900e+00
            +1.03339400e-03 * tc[1]
            -6.38038667e-08 * tc[2]
            -2.76168250e-11 * tc[3]
            +4.04270000e-15 * tc[4]
            +4.98497500e+04 * invT;
        /*species 13: CH3 */
        species[13] =
            +1.84405200e+00
            +3.06898700e-03 * tc[1]
            -7.43448333e-07 * tc[2]
            +9.46290250e-11 * tc[3]
            -4.90431800e-15 * tc[4]
            +1.64378100e+04 * invT;
        /*species 14: CH4 */
        species[14] =
            +6.83479000e-01
            +5.11862000e-03 * tc[1]
            -1.29170967e-06 * tc[2]
            +1.69639625e-10 * tc[3]
            -9.00684600e-15 * tc[4]
            -1.00807900e+04 * invT;
        /*species 15: CH3OH */
        species[15] =
            +3.02906100e+00
            +4.68829650e-03 * tc[1]
            -1.01675133e-06 * tc[2]
            +1.08969825e-10 * tc[3]
            -4.44944600e-15 * tc[4]
            -2.61579100e+04 * invT;
        /*species 17: CH2CO */
        species[17] =
            +5.03881700e+00
            +2.90242000e-03 * tc[1]
            -6.40318000e-07 * tc[2]
            +6.98621250e-11 * tc[3]
            -2.91773600e-15 * tc[4]
            -8.58340200e+03 * invT;
        /*species 20: C2H2 */
        species[20] =
            +3.43677000e+00
            +2.68801950e-03 * tc[1]
            -6.37605667e-07 * tc[2]
            +8.21594750e-11 * tc[3]
            -4.31342000e-15 * tc[4]
            +2.56676600e+04 * invT;
        /*species 22: C2H4 */
        species[22] =
            +2.52841900e+00
            +5.74259000e-03 * tc[1]
            -1.47279500e-06 * tc[2]
            +1.96115025e-10 * tc[3]
            -1.05336960e-14 * tc[4]
            +4.42828900e+03 * invT;
        /*species 23: CH3CO */
        species[23] =
            +4.61227900e+00
            +4.22494300e-03 * tc[1]
            -9.51382333e-07 * tc[2]
            +1.05959400e-10 * tc[3]
            -4.53680800e-15 * tc[4]
            -5.18786300e+03 * invT;
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
        /*species 16: C2H6 */
        species[16] =
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
        /*species 16: C2H6 */
        species[16] =
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
        /*species 18: HOCHO */
        species[18] =
            +4.35481850e-01
            +8.16815080e-03 * tc[1]
            -3.54191403e-06 * tc[2]
            +8.30332443e-10 * tc[3]
            -8.04352206e-14 * tc[4]
            -4.64616504e+04 * invT;
    } else {
        /*species 18: HOCHO */
        species[18] =
            +5.68733013e+00
            +2.57144684e-03 * tc[1]
            -6.07461710e-07 * tc[2]
            +7.24297908e-11 * tc[3]
            -3.41784398e-15 * tc[4]
            -4.83995400e+04 * invT;
    }

    /*species with midpoint at T=1385 kelvin */
    if (T < 1385) {
        /*species 19: CH3O2 */
        species[19] =
            +3.26146906e+00
            +5.04367995e-03 * tc[1]
            -1.07168728e-06 * tc[2]
            +5.23523168e-11 * tc[3]
            +8.36678206e-15 * tc[4]
            -6.84394259e+02 * invT;
    } else {
        /*species 19: CH3O2 */
        species[19] =
            +4.95787891e+00
            +3.95364313e-03 * tc[1]
            -8.94154113e-07 * tc[2]
            +1.03472834e-10 * tc[3]
            -4.78014660e-15 * tc[4]
            -1.53574838e+03 * invT;
    }

    /*species with midpoint at T=1388 kelvin */
    if (T < 1388) {
        /*species 21: C3H6 */
        species[21] =
            -6.05384556e-01
            +1.44553831e-02 * tc[1]
            -5.16289360e-06 * tc[2]
            +9.72035522e-10 * tc[3]
            -6.75780704e-14 * tc[4]
            +1.06688164e+03 * invT;
    } else {
        /*species 21: C3H6 */
        species[21] =
            +7.01595958e+00
            +6.85118170e-03 * tc[1]
            -1.55416578e-06 * tc[2]
            +1.80313601e-10 * tc[3]
            -8.34740252e-15 * tc[4]
            -1.87821271e+03 * invT;
    }

    /*species with midpoint at T=1400 kelvin */
    if (T < 1400) {
        /*species 24: C3H4XA */
        species[24] =
            +1.53983100e+00
            +8.16718500e-03 * tc[1]
            -5.88316667e-07 * tc[2]
            -1.16184125e-09 * tc[3]
            +3.45826200e-13 * tc[4]
            +2.25124300e+04 * invT;
    } else {
        /*species 24: C3H4XA */
        species[24] =
            +8.77625600e+00
            +2.65106900e-03 * tc[1]
            -1.23370600e-07 * tc[2]
            -7.56596500e-11 * tc[3]
            +1.01791620e-14 * tc[4]
            +1.95497200e+04 * invT;
    }

    /*species with midpoint at T=1397 kelvin */
    if (T < 1397) {
        /*species 25: C3H5XA */
        species[25] =
            -1.52913196e+00
            +1.67279550e-02 * tc[1]
            -8.44670090e-06 * tc[2]
            +2.57164385e-09 * tc[3]
            -3.46516680e-13 * tc[4]
            +1.93834226e+04 * invT;
    } else {
        /*species 25: C3H5XA */
        species[25] =
            +7.45883958e+00
            +5.63477415e-03 * tc[1]
            -1.27930955e-06 * tc[2]
            +1.48514780e-10 * tc[3]
            -6.87836060e-15 * tc[4]
            +1.63576092e+04 * invT;
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
    }

    /*species with midpoint at T=1382 kelvin */
    if (T < 1382) {
        /*species 30: C7H15X2 */
        species[30] =
            -1.03791558e+00
            +3.78363285e-02 * tc[1]
            -1.35824545e-05 * tc[2]
            +2.33169736e-09 * tc[3]
            -9.84721490e-14 * tc[4]
            -2.35605303e+03 * invT;
    } else {
        /*species 30: C7H15X2 */
        species[30] =
            +2.06368842e+01
            +1.61662402e-02 * tc[1]
            -3.64246023e-06 * tc[2]
            +4.20892650e-10 * tc[3]
            -1.94354818e-14 * tc[4]
            -1.05873616e+04 * invT;
    }

    /*species with midpoint at T=1391 kelvin */
    if (T < 1391) {
        /*species 31: NXC7H16 */
        species[31] =
            -2.26836187e+00
            +4.27177910e-02 * tc[1]
            -1.75115595e-05 * tc[2]
            +4.07364302e-09 * tc[3]
            -4.04789850e-13 * tc[4]
            -2.56586565e+04 * invT;
    } else {
        /*species 31: NXC7H16 */
        species[31] =
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
        /*species 12: CH2GSG */
        species[12] =
            +3.97126500e+00
            -8.49544500e-05 * tc[1]
            +3.41789667e-07 * tc[2]
            +6.23137750e-10 * tc[3]
            -3.96253200e-13 * tc[4]
            +4.98936800e+04 * invT;
        /*species 13: CH3 */
        species[13] =
            +2.43044300e+00
            +5.56205000e-03 * tc[1]
            -5.60073333e-06 * tc[2]
            +4.05457250e-09 * tc[3]
            -1.17299060e-12 * tc[4]
            +1.64237800e+04 * invT;
        /*species 14: CH4 */
        species[14] =
            +7.78741500e-01
            +8.73834000e-03 * tc[1]
            -9.27803000e-06 * tc[2]
            +7.62427000e-09 * tc[3]
            -2.44786200e-12 * tc[4]
            -9.82522900e+03 * invT;
        /*species 15: CH3OH */
        species[15] =
            +2.66011500e+00
            +3.67075400e-03 * tc[1]
            +2.39001700e-06 * tc[2]
            -2.19829850e-09 * tc[3]
            +4.78114000e-13 * tc[4]
            -2.53534800e+04 * invT;
        /*species 17: CH2CO */
        species[17] =
            +2.97497100e+00
            +6.05935500e-03 * tc[1]
            -7.81682000e-07 * tc[2]
            -1.61667125e-09 * tc[3]
            +7.81129800e-13 * tc[4]
            -7.63263700e+03 * invT;
        /*species 20: C2H2 */
        species[20] =
            +2.01356200e+00
            +7.59522500e-03 * tc[1]
            -5.38773000e-06 * tc[2]
            +2.26974800e-09 * tc[3]
            -3.82549200e-13 * tc[4]
            +2.61244400e+04 * invT;
        /*species 22: C2H4 */
        species[22] =
            -8.61488000e-01
            +1.39808150e-02 * tc[1]
            -1.12955900e-05 * tc[2]
            +6.96288000e-09 * tc[3]
            -1.94757580e-12 * tc[4]
            +5.57304600e+03 * invT;
        /*species 23: CH3CO */
        species[23] =
            +3.12527800e+00
            +4.88911000e-03 * tc[1]
            +1.50714933e-06 * tc[2]
            -2.25236550e-09 * tc[3]
            +6.38743600e-13 * tc[4]
            -4.10850800e+03 * invT;
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
        /*species 12: CH2GSG */
        species[12] =
            +3.55288900e+00
            +1.03339400e-03 * tc[1]
            -6.38038667e-08 * tc[2]
            -2.76168250e-11 * tc[3]
            +4.04270000e-15 * tc[4]
            +4.98497500e+04 * invT;
        /*species 13: CH3 */
        species[13] =
            +2.84405200e+00
            +3.06898700e-03 * tc[1]
            -7.43448333e-07 * tc[2]
            +9.46290250e-11 * tc[3]
            -4.90431800e-15 * tc[4]
            +1.64378100e+04 * invT;
        /*species 14: CH4 */
        species[14] =
            +1.68347900e+00
            +5.11862000e-03 * tc[1]
            -1.29170967e-06 * tc[2]
            +1.69639625e-10 * tc[3]
            -9.00684600e-15 * tc[4]
            -1.00807900e+04 * invT;
        /*species 15: CH3OH */
        species[15] =
            +4.02906100e+00
            +4.68829650e-03 * tc[1]
            -1.01675133e-06 * tc[2]
            +1.08969825e-10 * tc[3]
            -4.44944600e-15 * tc[4]
            -2.61579100e+04 * invT;
        /*species 17: CH2CO */
        species[17] =
            +6.03881700e+00
            +2.90242000e-03 * tc[1]
            -6.40318000e-07 * tc[2]
            +6.98621250e-11 * tc[3]
            -2.91773600e-15 * tc[4]
            -8.58340200e+03 * invT;
        /*species 20: C2H2 */
        species[20] =
            +4.43677000e+00
            +2.68801950e-03 * tc[1]
            -6.37605667e-07 * tc[2]
            +8.21594750e-11 * tc[3]
            -4.31342000e-15 * tc[4]
            +2.56676600e+04 * invT;
        /*species 22: C2H4 */
        species[22] =
            +3.52841900e+00
            +5.74259000e-03 * tc[1]
            -1.47279500e-06 * tc[2]
            +1.96115025e-10 * tc[3]
            -1.05336960e-14 * tc[4]
            +4.42828900e+03 * invT;
        /*species 23: CH3CO */
        species[23] =
            +5.61227900e+00
            +4.22494300e-03 * tc[1]
            -9.51382333e-07 * tc[2]
            +1.05959400e-10 * tc[3]
            -4.53680800e-15 * tc[4]
            -5.18786300e+03 * invT;
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
        /*species 16: C2H6 */
        species[16] =
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
        /*species 16: C2H6 */
        species[16] =
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
        /*species 18: HOCHO */
        species[18] =
            +1.43548185e+00
            +8.16815080e-03 * tc[1]
            -3.54191403e-06 * tc[2]
            +8.30332443e-10 * tc[3]
            -8.04352206e-14 * tc[4]
            -4.64616504e+04 * invT;
    } else {
        /*species 18: HOCHO */
        species[18] =
            +6.68733013e+00
            +2.57144684e-03 * tc[1]
            -6.07461710e-07 * tc[2]
            +7.24297908e-11 * tc[3]
            -3.41784398e-15 * tc[4]
            -4.83995400e+04 * invT;
    }

    /*species with midpoint at T=1385 kelvin */
    if (T < 1385) {
        /*species 19: CH3O2 */
        species[19] =
            +4.26146906e+00
            +5.04367995e-03 * tc[1]
            -1.07168728e-06 * tc[2]
            +5.23523168e-11 * tc[3]
            +8.36678206e-15 * tc[4]
            -6.84394259e+02 * invT;
    } else {
        /*species 19: CH3O2 */
        species[19] =
            +5.95787891e+00
            +3.95364313e-03 * tc[1]
            -8.94154113e-07 * tc[2]
            +1.03472834e-10 * tc[3]
            -4.78014660e-15 * tc[4]
            -1.53574838e+03 * invT;
    }

    /*species with midpoint at T=1388 kelvin */
    if (T < 1388) {
        /*species 21: C3H6 */
        species[21] =
            +3.94615444e-01
            +1.44553831e-02 * tc[1]
            -5.16289360e-06 * tc[2]
            +9.72035522e-10 * tc[3]
            -6.75780704e-14 * tc[4]
            +1.06688164e+03 * invT;
    } else {
        /*species 21: C3H6 */
        species[21] =
            +8.01595958e+00
            +6.85118170e-03 * tc[1]
            -1.55416578e-06 * tc[2]
            +1.80313601e-10 * tc[3]
            -8.34740252e-15 * tc[4]
            -1.87821271e+03 * invT;
    }

    /*species with midpoint at T=1400 kelvin */
    if (T < 1400) {
        /*species 24: C3H4XA */
        species[24] =
            +2.53983100e+00
            +8.16718500e-03 * tc[1]
            -5.88316667e-07 * tc[2]
            -1.16184125e-09 * tc[3]
            +3.45826200e-13 * tc[4]
            +2.25124300e+04 * invT;
    } else {
        /*species 24: C3H4XA */
        species[24] =
            +9.77625600e+00
            +2.65106900e-03 * tc[1]
            -1.23370600e-07 * tc[2]
            -7.56596500e-11 * tc[3]
            +1.01791620e-14 * tc[4]
            +1.95497200e+04 * invT;
    }

    /*species with midpoint at T=1397 kelvin */
    if (T < 1397) {
        /*species 25: C3H5XA */
        species[25] =
            -5.29131958e-01
            +1.67279550e-02 * tc[1]
            -8.44670090e-06 * tc[2]
            +2.57164385e-09 * tc[3]
            -3.46516680e-13 * tc[4]
            +1.93834226e+04 * invT;
    } else {
        /*species 25: C3H5XA */
        species[25] =
            +8.45883958e+00
            +5.63477415e-03 * tc[1]
            -1.27930955e-06 * tc[2]
            +1.48514780e-10 * tc[3]
            -6.87836060e-15 * tc[4]
            +1.63576092e+04 * invT;
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
    }

    /*species with midpoint at T=1382 kelvin */
    if (T < 1382) {
        /*species 30: C7H15X2 */
        species[30] =
            -3.79155767e-02
            +3.78363285e-02 * tc[1]
            -1.35824545e-05 * tc[2]
            +2.33169736e-09 * tc[3]
            -9.84721490e-14 * tc[4]
            -2.35605303e+03 * invT;
    } else {
        /*species 30: C7H15X2 */
        species[30] =
            +2.16368842e+01
            +1.61662402e-02 * tc[1]
            -3.64246023e-06 * tc[2]
            +4.20892650e-10 * tc[3]
            -1.94354818e-14 * tc[4]
            -1.05873616e+04 * invT;
    }

    /*species with midpoint at T=1391 kelvin */
    if (T < 1391) {
        /*species 31: NXC7H16 */
        species[31] =
            -1.26836187e+00
            +4.27177910e-02 * tc[1]
            -1.75115595e-05 * tc[2]
            +4.07364302e-09 * tc[3]
            -4.04789850e-13 * tc[4]
            -2.56586565e+04 * invT;
    } else {
        /*species 31: NXC7H16 */
        species[31] =
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
        /*species 12: CH2GSG */
        species[12] =
            +3.97126500e+00 * tc[0]
            -1.69908900e-04 * tc[1]
            +5.12684500e-07 * tc[2]
            +8.30850333e-10 * tc[3]
            -4.95316500e-13 * tc[4]
            +5.75320700e-02 ;
        /*species 13: CH3 */
        species[13] =
            +2.43044300e+00 * tc[0]
            +1.11241000e-02 * tc[1]
            -8.40110000e-06 * tc[2]
            +5.40609667e-09 * tc[3]
            -1.46623825e-12 * tc[4]
            +6.78979400e+00 ;
        /*species 14: CH4 */
        species[14] =
            +7.78741500e-01 * tc[0]
            +1.74766800e-02 * tc[1]
            -1.39170450e-05 * tc[2]
            +1.01656933e-08 * tc[3]
            -3.05982750e-12 * tc[4]
            +1.37221900e+01 ;
        /*species 15: CH3OH */
        species[15] =
            +2.66011500e+00 * tc[0]
            +7.34150800e-03 * tc[1]
            +3.58502550e-06 * tc[2]
            -2.93106467e-09 * tc[3]
            +5.97642500e-13 * tc[4]
            +1.12326300e+01 ;
        /*species 17: CH2CO */
        species[17] =
            +2.97497100e+00 * tc[0]
            +1.21187100e-02 * tc[1]
            -1.17252300e-06 * tc[2]
            -2.15556167e-09 * tc[3]
            +9.76412250e-13 * tc[4]
            +8.67355300e+00 ;
        /*species 20: C2H2 */
        species[20] =
            +2.01356200e+00 * tc[0]
            +1.51904500e-02 * tc[1]
            -8.08159500e-06 * tc[2]
            +3.02633067e-09 * tc[3]
            -4.78186500e-13 * tc[4]
            +8.80537800e+00 ;
        /*species 22: C2H4 */
        species[22] =
            -8.61488000e-01 * tc[0]
            +2.79616300e-02 * tc[1]
            -1.69433850e-05 * tc[2]
            +9.28384000e-09 * tc[3]
            -2.43446975e-12 * tc[4]
            +2.42114900e+01 ;
        /*species 23: CH3CO */
        species[23] =
            +3.12527800e+00 * tc[0]
            +9.77822000e-03 * tc[1]
            +2.26072400e-06 * tc[2]
            -3.00315400e-09 * tc[3]
            +7.98429500e-13 * tc[4]
            +1.12288500e+01 ;
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
        /*species 12: CH2GSG */
        species[12] =
            +3.55288900e+00 * tc[0]
            +2.06678800e-03 * tc[1]
            -9.57058000e-08 * tc[2]
            -3.68224333e-11 * tc[3]
            +5.05337500e-15 * tc[4]
            +1.68657000e+00 ;
        /*species 13: CH3 */
        species[13] =
            +2.84405200e+00 * tc[0]
            +6.13797400e-03 * tc[1]
            -1.11517250e-06 * tc[2]
            +1.26172033e-10 * tc[3]
            -6.13039750e-15 * tc[4]
            +5.45269700e+00 ;
        /*species 14: CH4 */
        species[14] =
            +1.68347900e+00 * tc[0]
            +1.02372400e-02 * tc[1]
            -1.93756450e-06 * tc[2]
            +2.26186167e-10 * tc[3]
            -1.12585575e-14 * tc[4]
            +9.62339500e+00 ;
        /*species 15: CH3OH */
        species[15] =
            +4.02906100e+00 * tc[0]
            +9.37659300e-03 * tc[1]
            -1.52512700e-06 * tc[2]
            +1.45293100e-10 * tc[3]
            -5.56180750e-15 * tc[4]
            +2.37819600e+00 ;
        /*species 17: CH2CO */
        species[17] =
            +6.03881700e+00 * tc[0]
            +5.80484000e-03 * tc[1]
            -9.60477000e-07 * tc[2]
            +9.31495000e-11 * tc[3]
            -3.64717000e-15 * tc[4]
            -7.65758100e+00 ;
        /*species 20: C2H2 */
        species[20] =
            +4.43677000e+00 * tc[0]
            +5.37603900e-03 * tc[1]
            -9.56408500e-07 * tc[2]
            +1.09545967e-10 * tc[3]
            -5.39177500e-15 * tc[4]
            -2.80033800e+00 ;
        /*species 22: C2H4 */
        species[22] =
            +3.52841900e+00 * tc[0]
            +1.14851800e-02 * tc[1]
            -2.20919250e-06 * tc[2]
            +2.61486700e-10 * tc[3]
            -1.31671200e-14 * tc[4]
            +2.23038900e+00 ;
        /*species 23: CH3CO */
        species[23] =
            +5.61227900e+00 * tc[0]
            +8.44988600e-03 * tc[1]
            -1.42707350e-06 * tc[2]
            +1.41279200e-10 * tc[3]
            -5.67101000e-15 * tc[4]
            -3.27494900e+00 ;
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
        /*species 16: C2H6 */
        species[16] =
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
        /*species 16: C2H6 */
        species[16] =
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
        /*species 18: HOCHO */
        species[18] =
            +1.43548185e+00 * tc[0]
            +1.63363016e-02 * tc[1]
            -5.31287105e-06 * tc[2]
            +1.10710992e-09 * tc[3]
            -1.00544026e-13 * tc[4]
            +1.72885798e+01 ;
    } else {
        /*species 18: HOCHO */
        species[18] =
            +6.68733013e+00 * tc[0]
            +5.14289368e-03 * tc[1]
            -9.11192565e-07 * tc[2]
            +9.65730543e-11 * tc[3]
            -4.27230498e-15 * tc[4]
            -1.13104798e+01 ;
    }

    /*species with midpoint at T=1385 kelvin */
    if (T < 1385) {
        /*species 19: CH3O2 */
        species[19] =
            +4.26146906e+00 * tc[0]
            +1.00873599e-02 * tc[1]
            -1.60753092e-06 * tc[2]
            +6.98030890e-11 * tc[3]
            +1.04584776e-14 * tc[4]
            +5.16330320e+00 ;
    } else {
        /*species 19: CH3O2 */
        species[19] =
            +5.95787891e+00 * tc[0]
            +7.90728626e-03 * tc[1]
            -1.34123117e-06 * tc[2]
            +1.37963779e-10 * tc[3]
            -5.97518325e-15 * tc[4]
            -4.71963886e+00 ;
    }

    /*species with midpoint at T=1388 kelvin */
    if (T < 1388) {
        /*species 21: C3H6 */
        species[21] =
            +3.94615444e-01 * tc[0]
            +2.89107662e-02 * tc[1]
            -7.74434040e-06 * tc[2]
            +1.29604736e-09 * tc[3]
            -8.44725880e-14 * tc[4]
            +2.19003736e+01 ;
    } else {
        /*species 21: C3H6 */
        species[21] =
            +8.01595958e+00 * tc[0]
            +1.37023634e-02 * tc[1]
            -2.33124867e-06 * tc[2]
            +2.40418134e-10 * tc[3]
            -1.04342532e-14 * tc[4]
            -2.00160668e+01 ;
    }

    /*species with midpoint at T=1400 kelvin */
    if (T < 1400) {
        /*species 24: C3H4XA */
        species[24] =
            +2.53983100e+00 * tc[0]
            +1.63343700e-02 * tc[1]
            -8.82475000e-07 * tc[2]
            -1.54912167e-09 * tc[3]
            +4.32282750e-13 * tc[4]
            +9.93570200e+00 ;
    } else {
        /*species 24: C3H4XA */
        species[24] =
            +9.77625600e+00 * tc[0]
            +5.30213800e-03 * tc[1]
            -1.85055900e-07 * tc[2]
            -1.00879533e-10 * tc[3]
            +1.27239525e-14 * tc[4]
            -3.07706100e+01 ;
    }

    /*species with midpoint at T=1397 kelvin */
    if (T < 1397) {
        /*species 25: C3H5XA */
        species[25] =
            -5.29131958e-01 * tc[0]
            +3.34559100e-02 * tc[1]
            -1.26700514e-05 * tc[2]
            +3.42885847e-09 * tc[3]
            -4.33145850e-13 * tc[4]
            +2.53067131e+01 ;
    } else {
        /*species 25: C3H5XA */
        species[25] =
            +8.45883958e+00 * tc[0]
            +1.12695483e-02 * tc[1]
            -1.91896432e-06 * tc[2]
            +1.98019706e-10 * tc[3]
            -8.59795075e-15 * tc[4]
            -2.25809450e+01 ;
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
    }

    /*species with midpoint at T=1382 kelvin */
    if (T < 1382) {
        /*species 30: C7H15X2 */
        species[30] =
            -3.79155767e-02 * tc[0]
            +7.56726570e-02 * tc[1]
            -2.03736817e-05 * tc[2]
            +3.10892981e-09 * tc[3]
            -1.23090186e-13 * tc[4]
            +3.37321506e+01 ;
    } else {
        /*species 30: C7H15X2 */
        species[30] =
            +2.16368842e+01 * tc[0]
            +3.23324804e-02 * tc[1]
            -5.46369035e-06 * tc[2]
            +5.61190200e-10 * tc[3]
            -2.42943523e-14 * tc[4]
            -8.52209653e+01 ;
    }

    /*species with midpoint at T=1391 kelvin */
    if (T < 1391) {
        /*species 31: NXC7H16 */
        species[31] =
            -1.26836187e+00 * tc[0]
            +8.54355820e-02 * tc[1]
            -2.62673393e-05 * tc[2]
            +5.43152403e-09 * tc[3]
            -5.05987313e-13 * tc[4]
            +3.53732912e+01 ;
    } else {
        /*species 31: NXC7H16 */
        species[31] =
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
    *LENIMC = 130;}


void egtransetLENRMC(int* LENRMC ) {
    *LENRMC = 20576;}


void egtransetNO(int* NO ) {
    *NO = 4;}


void egtransetKK(int* KK ) {
    *KK = 32;}


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
    WT[12] = 1.40270900E+01;
    WT[13] = 1.50350600E+01;
    WT[14] = 1.60430300E+01;
    WT[15] = 3.20424300E+01;
    WT[16] = 3.00701200E+01;
    WT[17] = 4.20376400E+01;
    WT[18] = 4.60258900E+01;
    WT[19] = 4.70338600E+01;
    WT[20] = 2.60382400E+01;
    WT[21] = 4.20812700E+01;
    WT[22] = 2.80541800E+01;
    WT[23] = 4.30456100E+01;
    WT[24] = 4.00653300E+01;
    WT[25] = 4.10733000E+01;
    WT[26] = 7.50880400E+01;
    WT[27] = 5.40924200E+01;
    WT[28] = 5.51003900E+01;
    WT[29] = 5.61083600E+01;
    WT[30] = 9.91976000E+01;
    WT[31] = 1.00205570E+02;
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
    EPS[13] = 1.44000000E+02;
    EPS[14] = 1.41400000E+02;
    EPS[15] = 4.81800000E+02;
    EPS[16] = 2.47500000E+02;
    EPS[17] = 4.36000000E+02;
    EPS[18] = 4.36000000E+02;
    EPS[19] = 4.81800000E+02;
    EPS[20] = 2.65300000E+02;
    EPS[21] = 3.07800000E+02;
    EPS[22] = 2.38400000E+02;
    EPS[23] = 4.36000000E+02;
    EPS[24] = 3.24800000E+02;
    EPS[25] = 3.16000000E+02;
    EPS[26] = 4.81500000E+02;
    EPS[27] = 3.57000000E+02;
    EPS[28] = 3.55000000E+02;
    EPS[29] = 3.55000000E+02;
    EPS[30] = 4.59600000E+02;
    EPS[31] = 4.59600000E+02;
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
    SIG[13] = 3.80000000E+00;
    SIG[14] = 3.74600000E+00;
    SIG[15] = 3.62600000E+00;
    SIG[16] = 4.35000000E+00;
    SIG[17] = 3.97000000E+00;
    SIG[18] = 3.97000000E+00;
    SIG[19] = 3.62600000E+00;
    SIG[20] = 3.72100000E+00;
    SIG[21] = 4.14000000E+00;
    SIG[22] = 3.49600000E+00;
    SIG[23] = 3.97000000E+00;
    SIG[24] = 4.29000000E+00;
    SIG[25] = 4.22000000E+00;
    SIG[26] = 4.99700000E+00;
    SIG[27] = 4.72000000E+00;
    SIG[28] = 4.65000000E+00;
    SIG[29] = 4.65000000E+00;
    SIG[30] = 6.25300000E+00;
    SIG[31] = 6.25300000E+00;
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
    POL[13] = 0.00000000E+00;
    POL[14] = 2.60000000E+00;
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
    ZROT[13] = 0.00000000E+00;
    ZROT[14] = 1.30000000E+01;
    ZROT[15] = 1.00000000E+00;
    ZROT[16] = 1.50000000E+00;
    ZROT[17] = 2.00000000E+00;
    ZROT[18] = 2.00000000E+00;
    ZROT[19] = 1.00000000E+00;
    ZROT[20] = 2.50000000E+00;
    ZROT[21] = 1.00000000E+00;
    ZROT[22] = 1.50000000E+00;
    ZROT[23] = 2.00000000E+00;
    ZROT[24] = 1.00000000E+00;
    ZROT[25] = 1.00000000E+00;
    ZROT[26] = 1.00000000E+00;
    ZROT[27] = 1.00000000E+00;
    ZROT[28] = 1.00000000E+00;
    ZROT[29] = 1.00000000E+00;
    ZROT[30] = 1.00000000E+00;
    ZROT[31] = 1.00000000E+00;
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
    NLIN[13] = 1;
    NLIN[14] = 2;
    NLIN[15] = 2;
    NLIN[16] = 2;
    NLIN[17] = 2;
    NLIN[18] = 2;
    NLIN[19] = 2;
    NLIN[20] = 1;
    NLIN[21] = 2;
    NLIN[22] = 2;
    NLIN[23] = 2;
    NLIN[24] = 1;
    NLIN[25] = 2;
    NLIN[26] = 2;
    NLIN[27] = 2;
    NLIN[28] = 2;
    NLIN[29] = 2;
    NLIN[30] = 2;
    NLIN[31] = 2;
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
    COFETA[48] = -2.02663469E+01;
    COFETA[49] = 3.63241793E+00;
    COFETA[50] = -3.95581049E-01;
    COFETA[51] = 1.74725495E-02;
    COFETA[52] = -2.02316497E+01;
    COFETA[53] = 3.63241793E+00;
    COFETA[54] = -3.95581049E-01;
    COFETA[55] = 1.74725495E-02;
    COFETA[56] = -2.00094664E+01;
    COFETA[57] = 3.57220167E+00;
    COFETA[58] = -3.87936446E-01;
    COFETA[59] = 1.71483254E-02;
    COFETA[60] = -2.05644525E+01;
    COFETA[61] = 3.03946431E+00;
    COFETA[62] = -2.16994867E-01;
    COFETA[63] = 5.61394012E-03;
    COFETA[64] = -2.45432160E+01;
    COFETA[65] = 5.15878990E+00;
    COFETA[66] = -5.75274341E-01;
    COFETA[67] = 2.44975136E-02;
    COFETA[68] = -2.23395647E+01;
    COFETA[69] = 3.86433912E+00;
    COFETA[70] = -3.41553983E-01;
    COFETA[71] = 1.17083447E-02;
    COFETA[72] = -2.22942453E+01;
    COFETA[73] = 3.86433912E+00;
    COFETA[74] = -3.41553983E-01;
    COFETA[75] = 1.17083447E-02;
    COFETA[76] = -2.03725491E+01;
    COFETA[77] = 3.03946431E+00;
    COFETA[78] = -2.16994867E-01;
    COFETA[79] = 5.61394012E-03;
    COFETA[80] = -2.47697856E+01;
    COFETA[81] = 5.30039568E+00;
    COFETA[82] = -5.89273639E-01;
    COFETA[83] = 2.49261407E-02;
    COFETA[84] = -2.49727893E+01;
    COFETA[85] = 5.27067543E+00;
    COFETA[86] = -5.71909526E-01;
    COFETA[87] = 2.36230940E-02;
    COFETA[88] = -2.39690472E+01;
    COFETA[89] = 5.11436059E+00;
    COFETA[90] = -5.71999954E-01;
    COFETA[91] = 2.44581334E-02;
    COFETA[92] = -2.23277173E+01;
    COFETA[93] = 3.86433912E+00;
    COFETA[94] = -3.41553983E-01;
    COFETA[95] = 1.17083447E-02;
    COFETA[96] = -2.50199983E+01;
    COFETA[97] = 5.20184077E+00;
    COFETA[98] = -5.57265947E-01;
    COFETA[99] = 2.27565676E-02;
    COFETA[100] = -2.50402232E+01;
    COFETA[101] = 5.25451220E+00;
    COFETA[102] = -5.67228955E-01;
    COFETA[103] = 2.33156489E-02;
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
    COFETA[120] = -2.19891717E+01;
    COFETA[121] = 3.46341268E+00;
    COFETA[122] = -2.80516687E-01;
    COFETA[123] = 8.70427548E-03;
    COFETA[124] = -2.19841167E+01;
    COFETA[125] = 3.46341268E+00;
    COFETA[126] = -2.80516687E-01;
    COFETA[127] = 8.70427548E-03;
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
    COFLAM[48] = 1.83932434E+01;
    COFLAM[49] = -6.25754971E+00;
    COFLAM[50] = 1.09378595E+00;
    COFLAM[51] = -5.49435895E-02;
    COFLAM[52] = 1.36305859E+01;
    COFLAM[53] = -4.45086721E+00;
    COFLAM[54] = 8.75862692E-01;
    COFLAM[55] = -4.60275277E-02;
    COFLAM[56] = 1.29622480E+01;
    COFLAM[57] = -4.85747192E+00;
    COFLAM[58] = 1.02918185E+00;
    COFLAM[59] = -5.69931976E-02;
    COFLAM[60] = -3.48612719E+00;
    COFLAM[61] = 1.33821415E+00;
    COFLAM[62] = 2.29051402E-01;
    COFLAM[63] = -2.22522544E-02;
    COFLAM[64] = -1.44773293E+01;
    COFLAM[65] = 6.20799727E+00;
    COFLAM[66] = -4.66686188E-01;
    COFLAM[67] = 1.03037078E-02;
    COFLAM[68] = -9.55248736E+00;
    COFLAM[69] = 4.54181017E+00;
    COFLAM[70] = -3.09443018E-01;
    COFLAM[71] = 5.98150058E-03;
    COFLAM[72] = -1.42688121E+01;
    COFLAM[73] = 6.13139162E+00;
    COFLAM[74] = -4.81580164E-01;
    COFLAM[75] = 1.17158883E-02;
    COFLAM[76] = 5.28056570E+00;
    COFLAM[77] = -1.92758693E+00;
    COFLAM[78] = 6.29141169E-01;
    COFLAM[79] = -3.87203309E-02;
    COFLAM[80] = -9.20687365E+00;
    COFLAM[81] = 5.13028609E+00;
    COFLAM[82] = -4.67868863E-01;
    COFLAM[83] = 1.64674383E-02;
    COFLAM[84] = -1.54410770E+01;
    COFLAM[85] = 6.67114766E+00;
    COFLAM[86] = -5.37137624E-01;
    COFLAM[87] = 1.38051704E-02;
    COFLAM[88] = -1.34447168E+01;
    COFLAM[89] = 6.12380203E+00;
    COFLAM[90] = -4.86657425E-01;
    COFLAM[91] = 1.24614518E-02;
    COFLAM[92] = -7.51756469E+00;
    COFLAM[93] = 3.30311461E+00;
    COFLAM[94] = -8.47654747E-02;
    COFLAM[95] = -6.42328466E-03;
    COFLAM[96] = -1.32966554E+01;
    COFLAM[97] = 5.92585034E+00;
    COFLAM[98] = -4.64901365E-01;
    COFLAM[99] = 1.16662523E-02;
    COFLAM[100] = -2.08328673E+01;
    COFLAM[101] = 9.07593204E+00;
    COFLAM[102] = -8.93990863E-01;
    COFLAM[103] = 3.11142957E-02;
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
    COFLAM[120] = -1.72838492E+01;
    COFLAM[121] = 6.97737723E+00;
    COFLAM[122] = -5.47365626E-01;
    COFLAM[123] = 1.30795303E-02;
    COFLAM[124] = -1.79582416E+01;
    COFLAM[125] = 7.27686902E+00;
    COFLAM[126] = -5.88898453E-01;
    COFLAM[127] = 1.49980279E-02;
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
    COFD[48] = -1.59404882E+01;
    COFD[49] = 3.66853818E+00;
    COFD[50] = -2.64346221E-01;
    COFD[51] = 1.15784613E-02;
    COFD[52] = -1.59633387E+01;
    COFD[53] = 3.66853818E+00;
    COFD[54] = -2.64346221E-01;
    COFD[55] = 1.15784613E-02;
    COFD[56] = -1.59327297E+01;
    COFD[57] = 3.65620899E+00;
    COFD[58] = -2.62933804E-01;
    COFD[59] = 1.15253223E-02;
    COFD[60] = -2.03844252E+01;
    COFD[61] = 5.18856872E+00;
    COFD[62] = -4.50001829E-01;
    COFD[63] = 1.91636142E-02;
    COFD[64] = -1.82673770E+01;
    COFD[65] = 4.39538102E+00;
    COFD[66] = -3.56367230E-01;
    COFD[67] = 1.54788461E-02;
    COFD[68] = -2.02646611E+01;
    COFD[69] = 5.10426133E+00;
    COFD[70] = -4.41256919E-01;
    COFD[71] = 1.88737290E-02;
    COFD[72] = -2.02822946E+01;
    COFD[73] = 5.10426133E+00;
    COFD[74] = -4.41256919E-01;
    COFD[75] = 1.88737290E-02;
    COFD[76] = -2.04649069E+01;
    COFD[77] = 5.18856872E+00;
    COFD[78] = -4.50001829E-01;
    COFD[79] = 1.91636142E-02;
    COFD[80] = -1.83039618E+01;
    COFD[81] = 4.47952077E+00;
    COFD[82] = -3.66569471E-01;
    COFD[83] = 1.58916129E-02;
    COFD[84] = -1.90859283E+01;
    COFD[85] = 4.68079396E+00;
    COFD[86] = -3.91231550E-01;
    COFD[87] = 1.69021170E-02;
    COFD[88] = -1.78815889E+01;
    COFD[89] = 4.34347890E+00;
    COFD[90] = -3.49890003E-01;
    COFD[91] = 1.52083459E-02;
    COFD[92] = -2.02693653E+01;
    COFD[93] = 5.10426133E+00;
    COFD[94] = -4.41256919E-01;
    COFD[95] = 1.88737290E-02;
    COFD[96] = -1.92783884E+01;
    COFD[97] = 4.73660584E+00;
    COFD[98] = -3.97704978E-01;
    COFD[99] = 1.71514887E-02;
    COFD[100] = -1.91796663E+01;
    COFD[101] = 4.70714822E+00;
    COFD[102] = -3.94261134E-01;
    COFD[103] = 1.70175169E-02;
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
    COFD[120] = -2.10674485E+01;
    COFD[121] = 5.15027524E+00;
    COFD[122] = -4.46126111E-01;
    COFD[123] = 1.90401391E-02;
    COFD[124] = -2.10685573E+01;
    COFD[125] = 5.15027524E+00;
    COFD[126] = -4.46126111E-01;
    COFD[127] = 1.90401391E-02;
    COFD[128] = -1.40756935E+01;
    COFD[129] = 3.07549274E+00;
    COFD[130] = -1.88889344E-01;
    COFD[131] = 8.37152866E-03;
    COFD[132] = -1.32093628E+01;
    COFD[133] = 2.90778936E+00;
    COFD[134] = -1.67388544E-01;
    COFD[135] = 7.45220609E-03;
    COFD[136] = -1.09595712E+01;
    COFD[137] = 2.30836460E+00;
    COFD[138] = -8.76339315E-02;
    COFD[139] = 3.90878445E-03;
    COFD[140] = -1.34230272E+01;
    COFD[141] = 3.48624238E+00;
    COFD[142] = -2.41554467E-01;
    COFD[143] = 1.06263545E-02;
    COFD[144] = -1.32244035E+01;
    COFD[145] = 2.90778936E+00;
    COFD[146] = -1.67388544E-01;
    COFD[147] = 7.45220609E-03;
    COFD[148] = -1.94093572E+01;
    COFD[149] = 5.16013126E+00;
    COFD[150] = -4.46824543E-01;
    COFD[151] = 1.90464887E-02;
    COFD[152] = -1.43139231E+01;
    COFD[153] = 3.17651319E+00;
    COFD[154] = -2.02028974E-01;
    COFD[155] = 8.94232502E-03;
    COFD[156] = -1.43190389E+01;
    COFD[157] = 3.17651319E+00;
    COFD[158] = -2.02028974E-01;
    COFD[159] = 8.94232502E-03;
    COFD[160] = -1.43238998E+01;
    COFD[161] = 3.17651319E+00;
    COFD[162] = -2.02028974E-01;
    COFD[163] = 8.94232502E-03;
    COFD[164] = -1.70534856E+01;
    COFD[165] = 4.14240922E+00;
    COFD[166] = -3.25239774E-01;
    COFD[167] = 1.41980687E-02;
    COFD[168] = -1.40999008E+01;
    COFD[169] = 3.08120012E+00;
    COFD[170] = -1.89629903E-01;
    COFD[171] = 8.40361952E-03;
    COFD[172] = -1.94373127E+01;
    COFD[173] = 5.02567894E+00;
    COFD[174] = -4.32045169E-01;
    COFD[175] = 1.85132214E-02;
    COFD[176] = -1.50584249E+01;
    COFD[177] = 3.47945612E+00;
    COFD[178] = -2.40703722E-01;
    COFD[179] = 1.05907441E-02;
    COFD[180] = -1.50766130E+01;
    COFD[181] = 3.47945612E+00;
    COFD[182] = -2.40703722E-01;
    COFD[183] = 1.05907441E-02;
    COFD[184] = -1.50270339E+01;
    COFD[185] = 3.46140064E+00;
    COFD[186] = -2.38440092E-01;
    COFD[187] = 1.04960087E-02;
    COFD[188] = -1.93364585E+01;
    COFD[189] = 4.98286777E+00;
    COFD[190] = -4.26970814E-01;
    COFD[191] = 1.83122917E-02;
    COFD[192] = -1.72112971E+01;
    COFD[193] = 4.15807461E+00;
    COFD[194] = -3.27178539E-01;
    COFD[195] = 1.42784349E-02;
    COFD[196] = -1.90883268E+01;
    COFD[197] = 4.84384483E+00;
    COFD[198] = -4.10265575E-01;
    COFD[199] = 1.76414287E-02;
    COFD[200] = -1.91004157E+01;
    COFD[201] = 4.84384483E+00;
    COFD[202] = -4.10265575E-01;
    COFD[203] = 1.76414287E-02;
    COFD[204] = -1.93925667E+01;
    COFD[205] = 4.98286777E+00;
    COFD[206] = -4.26970814E-01;
    COFD[207] = 1.83122917E-02;
    COFD[208] = -1.72286007E+01;
    COFD[209] = 4.24084025E+00;
    COFD[210] = -3.37428619E-01;
    COFD[211] = 1.47032793E-02;
    COFD[212] = -1.79361160E+01;
    COFD[213] = 4.42139452E+00;
    COFD[214] = -3.59567329E-01;
    COFD[215] = 1.56103969E-02;
    COFD[216] = -1.68343393E+01;
    COFD[217] = 4.11954900E+00;
    COFD[218] = -3.22470391E-01;
    COFD[219] = 1.40859564E-02;
    COFD[220] = -1.90915649E+01;
    COFD[221] = 4.84384483E+00;
    COFD[222] = -4.10265575E-01;
    COFD[223] = 1.76414287E-02;
    COFD[224] = -1.81499793E+01;
    COFD[225] = 4.48398491E+00;
    COFD[226] = -3.67097129E-01;
    COFD[227] = 1.59123634E-02;
    COFD[228] = -1.80480958E+01;
    COFD[229] = 4.45434023E+00;
    COFD[230] = -3.63584633E-01;
    COFD[231] = 1.57739270E-02;
    COFD[232] = -1.98296243E+01;
    COFD[233] = 4.98207523E+00;
    COFD[234] = -4.26877291E-01;
    COFD[235] = 1.83086094E-02;
    COFD[236] = -1.86652603E+01;
    COFD[237] = 4.61260432E+00;
    COFD[238] = -3.82854484E-01;
    COFD[239] = 1.65575163E-02;
    COFD[240] = -1.86234701E+01;
    COFD[241] = 4.60336076E+00;
    COFD[242] = -3.81691643E-01;
    COFD[243] = 1.65085234E-02;
    COFD[244] = -1.86254955E+01;
    COFD[245] = 4.60336076E+00;
    COFD[246] = -3.81691643E-01;
    COFD[247] = 1.65085234E-02;
    COFD[248] = -1.99785176E+01;
    COFD[249] = 4.92184026E+00;
    COFD[250] = -4.19745472E-01;
    COFD[251] = 1.80268154E-02;
    COFD[252] = -1.99792167E+01;
    COFD[253] = 4.92184026E+00;
    COFD[254] = -4.19745472E-01;
    COFD[255] = 1.80268154E-02;
    COFD[256] = -1.16906297E+01;
    COFD[257] = 2.47469981E+00;
    COFD[258] = -1.10436257E-01;
    COFD[259] = 4.95273813E-03;
    COFD[260] = -1.09595712E+01;
    COFD[261] = 2.30836460E+00;
    COFD[262] = -8.76339315E-02;
    COFD[263] = 3.90878445E-03;
    COFD[264] = -1.03270606E+01;
    COFD[265] = 2.19285409E+00;
    COFD[266] = -7.54492786E-02;
    COFD[267] = 3.51398213E-03;
    COFD[268] = -1.14366381E+01;
    COFD[269] = 2.78323501E+00;
    COFD[270] = -1.51214064E-01;
    COFD[271] = 6.75150012E-03;
    COFD[272] = -1.09628982E+01;
    COFD[273] = 2.30836460E+00;
    COFD[274] = -8.76339315E-02;
    COFD[275] = 3.90878445E-03;
    COFD[276] = -1.71982995E+01;
    COFD[277] = 4.63881404E+00;
    COFD[278] = -3.86139633E-01;
    COFD[279] = 1.66955081E-02;
    COFD[280] = -1.18988955E+01;
    COFD[281] = 2.57507000E+00;
    COFD[282] = -1.24033737E-01;
    COFD[283] = 5.56694959E-03;
    COFD[284] = -1.18998012E+01;
    COFD[285] = 2.57507000E+00;
    COFD[286] = -1.24033737E-01;
    COFD[287] = 5.56694959E-03;
    COFD[288] = -1.19006548E+01;
    COFD[289] = 2.57507000E+00;
    COFD[290] = -1.24033737E-01;
    COFD[291] = 5.56694959E-03;
    COFD[292] = -1.37794315E+01;
    COFD[293] = 3.23973858E+00;
    COFD[294] = -2.09989036E-01;
    COFD[295] = 9.27667906E-03;
    COFD[296] = -1.17159737E+01;
    COFD[297] = 2.48123210E+00;
    COFD[298] = -1.11322604E-01;
    COFD[299] = 4.99282389E-03;
    COFD[300] = -1.60528285E+01;
    COFD[301] = 4.11188603E+00;
    COFD[302] = -3.21540884E-01;
    COFD[303] = 1.40482564E-02;
    COFD[304] = -1.25098960E+01;
    COFD[305] = 2.77873601E+00;
    COFD[306] = -1.50637360E-01;
    COFD[307] = 6.72684281E-03;
    COFD[308] = -1.25141260E+01;
    COFD[309] = 2.77873601E+00;
    COFD[310] = -1.50637360E-01;
    COFD[311] = 6.72684281E-03;
    COFD[312] = -1.24693568E+01;
    COFD[313] = 2.76686648E+00;
    COFD[314] = -1.49120141E-01;
    COFD[315] = 6.66220432E-03;
    COFD[316] = -1.59537247E+01;
    COFD[317] = 4.07051484E+00;
    COFD[318] = -3.16303109E-01;
    COFD[319] = 1.38259377E-02;
    COFD[320] = -1.39658996E+01;
    COFD[321] = 3.24966086E+00;
    COFD[322] = -2.11199992E-01;
    COFD[323] = 9.32580661E-03;
    COFD[324] = -1.57034851E+01;
    COFD[325] = 3.93614244E+00;
    COFD[326] = -2.99111497E-01;
    COFD[327] = 1.30888229E-02;
    COFD[328] = -1.57054717E+01;
    COFD[329] = 3.93614244E+00;
    COFD[330] = -2.99111497E-01;
    COFD[331] = 1.30888229E-02;
    COFD[332] = -1.59632479E+01;
    COFD[333] = 4.07051484E+00;
    COFD[334] = -3.16303109E-01;
    COFD[335] = 1.38259377E-02;
    COFD[336] = -1.39315266E+01;
    COFD[337] = 3.30394764E+00;
    COFD[338] = -2.17920112E-01;
    COFD[339] = 9.60284243E-03;
    COFD[340] = -1.45715797E+01;
    COFD[341] = 3.49477850E+00;
    COFD[342] = -2.42635772E-01;
    COFD[343] = 1.06721490E-02;
    COFD[344] = -1.36336373E+01;
    COFD[345] = 3.22088176E+00;
    COFD[346] = -2.07623790E-01;
    COFD[347] = 9.17771542E-03;
    COFD[348] = -1.57040212E+01;
    COFD[349] = 3.93614244E+00;
    COFD[350] = -2.99111497E-01;
    COFD[351] = 1.30888229E-02;
    COFD[352] = -1.47725694E+01;
    COFD[353] = 3.55444478E+00;
    COFD[354] = -2.50272707E-01;
    COFD[355] = 1.09990787E-02;
    COFD[356] = -1.46719197E+01;
    COFD[357] = 3.52400594E+00;
    COFD[358] = -2.46379985E-01;
    COFD[359] = 1.08326032E-02;
    COFD[360] = -1.64819183E+01;
    COFD[361] = 4.11726215E+00;
    COFD[362] = -3.22193015E-01;
    COFD[363] = 1.40747074E-02;
    COFD[364] = -1.51448279E+01;
    COFD[365] = 3.64565939E+00;
    COFD[366] = -2.61726871E-01;
    COFD[367] = 1.14799244E-02;
    COFD[368] = -1.51159870E+01;
    COFD[369] = 3.64206330E+00;
    COFD[370] = -2.61313444E-01;
    COFD[371] = 1.14642754E-02;
    COFD[372] = -1.51163041E+01;
    COFD[373] = 3.64206330E+00;
    COFD[374] = -2.61313444E-01;
    COFD[375] = 1.14642754E-02;
    COFD[376] = -1.64898528E+01;
    COFD[377] = 4.01175649E+00;
    COFD[378] = -3.08860971E-01;
    COFD[379] = 1.35100076E-02;
    COFD[380] = -1.64899530E+01;
    COFD[381] = 4.01175649E+00;
    COFD[382] = -3.08860971E-01;
    COFD[383] = 1.35100076E-02;
    COFD[384] = -1.42894441E+01;
    COFD[385] = 3.67490723E+00;
    COFD[386] = -2.65114792E-01;
    COFD[387] = 1.16092671E-02;
    COFD[388] = -1.34230272E+01;
    COFD[389] = 3.48624238E+00;
    COFD[390] = -2.41554467E-01;
    COFD[391] = 1.06263545E-02;
    COFD[392] = -1.14366381E+01;
    COFD[393] = 2.78323501E+00;
    COFD[394] = -1.51214064E-01;
    COFD[395] = 6.75150012E-03;
    COFD[396] = -1.47968712E+01;
    COFD[397] = 4.23027636E+00;
    COFD[398] = -3.36139991E-01;
    COFD[399] = 1.46507621E-02;
    COFD[400] = -1.34247866E+01;
    COFD[401] = 3.48624238E+00;
    COFD[402] = -2.41554467E-01;
    COFD[403] = 1.06263545E-02;
    COFD[404] = -1.95739570E+01;
    COFD[405] = 5.61113230E+00;
    COFD[406] = -4.90190187E-01;
    COFD[407] = 2.03260675E-02;
    COFD[408] = -1.46550083E+01;
    COFD[409] = 3.83606243E+00;
    COFD[410] = -2.86076532E-01;
    COFD[411] = 1.25205829E-02;
    COFD[412] = -1.46554748E+01;
    COFD[413] = 3.83606243E+00;
    COFD[414] = -2.86076532E-01;
    COFD[415] = 1.25205829E-02;
    COFD[416] = -1.46559141E+01;
    COFD[417] = 3.83606243E+00;
    COFD[418] = -2.86076532E-01;
    COFD[419] = 1.25205829E-02;
    COFD[420] = -1.76147026E+01;
    COFD[421] = 4.86049500E+00;
    COFD[422] = -4.12200578E-01;
    COFD[423] = 1.77160971E-02;
    COFD[424] = -1.43151174E+01;
    COFD[425] = 3.68038508E+00;
    COFD[426] = -2.65779346E-01;
    COFD[427] = 1.16360771E-02;
    COFD[428] = -1.97550088E+01;
    COFD[429] = 5.56931926E+00;
    COFD[430] = -4.89105511E-01;
    COFD[431] = 2.04493129E-02;
    COFD[432] = -1.57972369E+01;
    COFD[433] = 4.22225052E+00;
    COFD[434] = -3.35156428E-01;
    COFD[435] = 1.46104855E-02;
    COFD[436] = -1.57994893E+01;
    COFD[437] = 4.22225052E+00;
    COFD[438] = -3.35156428E-01;
    COFD[439] = 1.46104855E-02;
    COFD[440] = -1.57199037E+01;
    COFD[441] = 4.19936335E+00;
    COFD[442] = -3.32311009E-01;
    COFD[443] = 1.44921003E-02;
    COFD[444] = -1.96866103E+01;
    COFD[445] = 5.54637286E+00;
    COFD[446] = -4.87070324E-01;
    COFD[447] = 2.03983467E-02;
    COFD[448] = -1.78637178E+01;
    COFD[449] = 4.88268692E+00;
    COFD[450] = -4.14917638E-01;
    COFD[451] = 1.78274298E-02;
    COFD[452] = -1.94688688E+01;
    COFD[453] = 5.43830787E+00;
    COFD[454] = -4.75472880E-01;
    COFD[455] = 1.99909996E-02;
    COFD[456] = -1.94698843E+01;
    COFD[457] = 5.43830787E+00;
    COFD[458] = -4.75472880E-01;
    COFD[459] = 1.99909996E-02;
    COFD[460] = -1.96914944E+01;
    COFD[461] = 5.54637286E+00;
    COFD[462] = -4.87070324E-01;
    COFD[463] = 2.03983467E-02;
    COFD[464] = -1.79310765E+01;
    COFD[465] = 4.98037650E+00;
    COFD[466] = -4.26676911E-01;
    COFD[467] = 1.83007231E-02;
    COFD[468] = -1.85748546E+01;
    COFD[469] = 5.14789919E+00;
    COFD[470] = -4.45930850E-01;
    COFD[471] = 1.90363341E-02;
    COFD[472] = -1.74407963E+01;
    COFD[473] = 4.83580036E+00;
    COFD[474] = -4.09383573E-01;
    COFD[475] = 1.76098175E-02;
    COFD[476] = -1.94691430E+01;
    COFD[477] = 5.43830787E+00;
    COFD[478] = -4.75472880E-01;
    COFD[479] = 1.99909996E-02;
    COFD[480] = -1.87647862E+01;
    COFD[481] = 5.19146813E+00;
    COFD[482] = -4.50340408E-01;
    COFD[483] = 1.91768178E-02;
    COFD[484] = -1.86493112E+01;
    COFD[485] = 5.16040659E+00;
    COFD[486] = -4.46843492E-01;
    COFD[487] = 1.90466181E-02;
    COFD[488] = -2.01262921E+01;
    COFD[489] = 5.54581286E+00;
    COFD[490] = -4.87014004E-01;
    COFD[491] = 2.03965482E-02;
    COFD[492] = -1.92784178E+01;
    COFD[493] = 5.32291505E+00;
    COFD[494] = -4.65883522E-01;
    COFD[495] = 1.97916109E-02;
    COFD[496] = -1.92360228E+01;
    COFD[497] = 5.31542554E+00;
    COFD[498] = -4.65003780E-01;
    COFD[499] = 1.97570185E-02;
    COFD[500] = -1.92361841E+01;
    COFD[501] = 5.31542554E+00;
    COFD[502] = -4.65003780E-01;
    COFD[503] = 1.97570185E-02;
    COFD[504] = -2.03113704E+01;
    COFD[505] = 5.50136606E+00;
    COFD[506] = -4.82461887E-01;
    COFD[507] = 2.02471523E-02;
    COFD[508] = -2.03114210E+01;
    COFD[509] = 5.50136606E+00;
    COFD[510] = -4.82461887E-01;
    COFD[511] = 2.02471523E-02;
    COFD[512] = -1.40949196E+01;
    COFD[513] = 3.07549274E+00;
    COFD[514] = -1.88889344E-01;
    COFD[515] = 8.37152866E-03;
    COFD[516] = -1.32244035E+01;
    COFD[517] = 2.90778936E+00;
    COFD[518] = -1.67388544E-01;
    COFD[519] = 7.45220609E-03;
    COFD[520] = -1.09628982E+01;
    COFD[521] = 2.30836460E+00;
    COFD[522] = -8.76339315E-02;
    COFD[523] = 3.90878445E-03;
    COFD[524] = -1.34247866E+01;
    COFD[525] = 3.48624238E+00;
    COFD[526] = -2.41554467E-01;
    COFD[527] = 1.06263545E-02;
    COFD[528] = -1.32399106E+01;
    COFD[529] = 2.90778936E+00;
    COFD[530] = -1.67388544E-01;
    COFD[531] = 7.45220609E-03;
    COFD[532] = -1.94253036E+01;
    COFD[533] = 5.16013126E+00;
    COFD[534] = -4.46824543E-01;
    COFD[535] = 1.90464887E-02;
    COFD[536] = -1.43340796E+01;
    COFD[537] = 3.17651319E+00;
    COFD[538] = -2.02028974E-01;
    COFD[539] = 8.94232502E-03;
    COFD[540] = -1.43394069E+01;
    COFD[541] = 3.17651319E+00;
    COFD[542] = -2.02028974E-01;
    COFD[543] = 8.94232502E-03;
    COFD[544] = -1.43444709E+01;
    COFD[545] = 3.17651319E+00;
    COFD[546] = -2.02028974E-01;
    COFD[547] = 8.94232502E-03;
    COFD[548] = -1.70757047E+01;
    COFD[549] = 4.14240922E+00;
    COFD[550] = -3.25239774E-01;
    COFD[551] = 1.41980687E-02;
    COFD[552] = -1.41191261E+01;
    COFD[553] = 3.08120012E+00;
    COFD[554] = -1.89629903E-01;
    COFD[555] = 8.40361952E-03;
    COFD[556] = -1.94570287E+01;
    COFD[557] = 5.02567894E+00;
    COFD[558] = -4.32045169E-01;
    COFD[559] = 1.85132214E-02;
    COFD[560] = -1.50724636E+01;
    COFD[561] = 3.47945612E+00;
    COFD[562] = -2.40703722E-01;
    COFD[563] = 1.05907441E-02;
    COFD[564] = -1.50911794E+01;
    COFD[565] = 3.47945612E+00;
    COFD[566] = -2.40703722E-01;
    COFD[567] = 1.05907441E-02;
    COFD[568] = -1.50420953E+01;
    COFD[569] = 3.46140064E+00;
    COFD[570] = -2.38440092E-01;
    COFD[571] = 1.04960087E-02;
    COFD[572] = -1.93566243E+01;
    COFD[573] = 4.98286777E+00;
    COFD[574] = -4.26970814E-01;
    COFD[575] = 1.83122917E-02;
    COFD[576] = -1.72310232E+01;
    COFD[577] = 4.15807461E+00;
    COFD[578] = -3.27178539E-01;
    COFD[579] = 1.42784349E-02;
    COFD[580] = -1.91102652E+01;
    COFD[581] = 4.84384483E+00;
    COFD[582] = -4.10265575E-01;
    COFD[583] = 1.76414287E-02;
    COFD[584] = -1.91229033E+01;
    COFD[585] = 4.84384483E+00;
    COFD[586] = -4.10265575E-01;
    COFD[587] = 1.76414287E-02;
    COFD[588] = -1.94151822E+01;
    COFD[589] = 4.98286777E+00;
    COFD[590] = -4.26970814E-01;
    COFD[591] = 1.83122917E-02;
    COFD[592] = -1.72473011E+01;
    COFD[593] = 4.24084025E+00;
    COFD[594] = -3.37428619E-01;
    COFD[595] = 1.47032793E-02;
    COFD[596] = -1.79580609E+01;
    COFD[597] = 4.42139452E+00;
    COFD[598] = -3.59567329E-01;
    COFD[599] = 1.56103969E-02;
    COFD[600] = -1.68535757E+01;
    COFD[601] = 4.11954900E+00;
    COFD[602] = -3.22470391E-01;
    COFD[603] = 1.40859564E-02;
    COFD[604] = -1.91136491E+01;
    COFD[605] = 4.84384483E+00;
    COFD[606] = -4.10265575E-01;
    COFD[607] = 1.76414287E-02;
    COFD[608] = -1.81716176E+01;
    COFD[609] = 4.48398491E+00;
    COFD[610] = -3.67097129E-01;
    COFD[611] = 1.59123634E-02;
    COFD[612] = -1.80698901E+01;
    COFD[613] = 4.45434023E+00;
    COFD[614] = -3.63584633E-01;
    COFD[615] = 1.57739270E-02;
    COFD[616] = -1.98546695E+01;
    COFD[617] = 4.98207523E+00;
    COFD[618] = -4.26877291E-01;
    COFD[619] = 1.83086094E-02;
    COFD[620] = -1.86886689E+01;
    COFD[621] = 4.61260432E+00;
    COFD[622] = -3.82854484E-01;
    COFD[623] = 1.65575163E-02;
    COFD[624] = -1.86469792E+01;
    COFD[625] = 4.60336076E+00;
    COFD[626] = -3.81691643E-01;
    COFD[627] = 1.65085234E-02;
    COFD[628] = -1.86491023E+01;
    COFD[629] = 4.60336076E+00;
    COFD[630] = -3.81691643E-01;
    COFD[631] = 1.65085234E-02;
    COFD[632] = -2.00047095E+01;
    COFD[633] = 4.92184026E+00;
    COFD[634] = -4.19745472E-01;
    COFD[635] = 1.80268154E-02;
    COFD[636] = -2.00054461E+01;
    COFD[637] = 4.92184026E+00;
    COFD[638] = -4.19745472E-01;
    COFD[639] = 1.80268154E-02;
    COFD[640] = -2.10643259E+01;
    COFD[641] = 5.53614847E+00;
    COFD[642] = -4.86046736E-01;
    COFD[643] = 2.03659188E-02;
    COFD[644] = -1.94093572E+01;
    COFD[645] = 5.16013126E+00;
    COFD[646] = -4.46824543E-01;
    COFD[647] = 1.90464887E-02;
    COFD[648] = -1.71982995E+01;
    COFD[649] = 4.63881404E+00;
    COFD[650] = -3.86139633E-01;
    COFD[651] = 1.66955081E-02;
    COFD[652] = -1.95739570E+01;
    COFD[653] = 5.61113230E+00;
    COFD[654] = -4.90190187E-01;
    COFD[655] = 2.03260675E-02;
    COFD[656] = -1.94253036E+01;
    COFD[657] = 5.16013126E+00;
    COFD[658] = -4.46824543E-01;
    COFD[659] = 1.90464887E-02;
    COFD[660] = -1.19157919E+01;
    COFD[661] = 9.28955130E-01;
    COFD[662] = 2.42107090E-01;
    COFD[663] = -1.59823963E-02;
    COFD[664] = -2.12652533E+01;
    COFD[665] = 5.59961818E+00;
    COFD[666] = -4.91624858E-01;
    COFD[667] = 2.05035550E-02;
    COFD[668] = -2.06463744E+01;
    COFD[669] = 5.41688482E+00;
    COFD[670] = -4.73387188E-01;
    COFD[671] = 1.99280175E-02;
    COFD[672] = -2.06516336E+01;
    COFD[673] = 5.41688482E+00;
    COFD[674] = -4.73387188E-01;
    COFD[675] = 1.99280175E-02;
    COFD[676] = -2.07653719E+01;
    COFD[677] = 5.01092022E+00;
    COFD[678] = -3.77985635E-01;
    COFD[679] = 1.40968645E-02;
    COFD[680] = -2.11388331E+01;
    COFD[681] = 5.55529675E+00;
    COFD[682] = -4.87942518E-01;
    COFD[683] = 2.04249054E-02;
    COFD[684] = -1.77563250E+01;
    COFD[685] = 3.57475686E+00;
    COFD[686] = -1.56396297E-01;
    COFD[687] = 3.12157721E-03;
    COFD[688] = -2.12639214E+01;
    COFD[689] = 5.61184117E+00;
    COFD[690] = -4.90532156E-01;
    COFD[691] = 2.03507922E-02;
    COFD[692] = -2.12831323E+01;
    COFD[693] = 5.61184117E+00;
    COFD[694] = -4.90532156E-01;
    COFD[695] = 2.03507922E-02;
    COFD[696] = -2.14087397E+01;
    COFD[697] = 5.57282008E+00;
    COFD[698] = -4.76690890E-01;
    COFD[699] = 1.94000719E-02;
    COFD[700] = -1.80253664E+01;
    COFD[701] = 3.69199168E+00;
    COFD[702] = -1.74005516E-01;
    COFD[703] = 3.97694372E-03;
    COFD[704] = -2.13148887E+01;
    COFD[705] = 5.27210469E+00;
    COFD[706] = -4.21419216E-01;
    COFD[707] = 1.63567178E-02;
    COFD[708] = -1.87383952E+01;
    COFD[709] = 3.96926341E+00;
    COFD[710] = -2.16412264E-01;
    COFD[711] = 6.06012078E-03;
    COFD[712] = -1.87515645E+01;
    COFD[713] = 3.96926341E+00;
    COFD[714] = -2.16412264E-01;
    COFD[715] = 6.06012078E-03;
    COFD[716] = -1.80862867E+01;
    COFD[717] = 3.69199168E+00;
    COFD[718] = -1.74005516E-01;
    COFD[719] = 3.97694372E-03;
    COFD[720] = -2.09565916E+01;
    COFD[721] = 5.18380539E+00;
    COFD[722] = -4.06234719E-01;
    COFD[723] = 1.55515345E-02;
    COFD[724] = -2.06310304E+01;
    COFD[725] = 4.89289496E+00;
    COFD[726] = -3.59346263E-01;
    COFD[727] = 1.31570901E-02;
    COFD[728] = -2.11309197E+01;
    COFD[729] = 5.32644193E+00;
    COFD[730] = -4.30581064E-01;
    COFD[731] = 1.68379725E-02;
    COFD[732] = -1.87419199E+01;
    COFD[733] = 3.96926341E+00;
    COFD[734] = -2.16412264E-01;
    COFD[735] = 6.06012078E-03;
    COFD[736] = -2.04397451E+01;
    COFD[737] = 4.77398686E+00;
    COFD[738] = -3.40522956E-01;
    COFD[739] = 1.22072846E-02;
    COFD[740] = -2.05372411E+01;
    COFD[741] = 4.83379373E+00;
    COFD[742] = -3.50008083E-01;
    COFD[743] = 1.26863426E-02;
    COFD[744] = -1.73636900E+01;
    COFD[745] = 3.17377130E+00;
    COFD[746] = -1.00394383E-01;
    COFD[747] = 5.69083899E-04;
    COFD[748] = -2.02184916E+01;
    COFD[749] = 4.57152878E+00;
    COFD[750] = -3.08371263E-01;
    COFD[751] = 1.05838559E-02;
    COFD[752] = -2.02265558E+01;
    COFD[753] = 4.58441724E+00;
    COFD[754] = -3.10392854E-01;
    COFD[755] = 1.06849990E-02;
    COFD[756] = -2.02287739E+01;
    COFD[757] = 4.58441724E+00;
    COFD[758] = -3.10392854E-01;
    COFD[759] = 1.06849990E-02;
    COFD[760] = -1.91326792E+01;
    COFD[761] = 3.82263611E+00;
    COFD[762] = -1.93983472E-01;
    COFD[763] = 4.95789388E-03;
    COFD[764] = -1.91334529E+01;
    COFD[765] = 3.82263611E+00;
    COFD[766] = -1.93983472E-01;
    COFD[767] = 4.95789388E-03;
    COFD[768] = -1.52414485E+01;
    COFD[769] = 3.35922578E+00;
    COFD[770] = -2.25181399E-01;
    COFD[771] = 9.92132878E-03;
    COFD[772] = -1.43139231E+01;
    COFD[773] = 3.17651319E+00;
    COFD[774] = -2.02028974E-01;
    COFD[775] = 8.94232502E-03;
    COFD[776] = -1.18988955E+01;
    COFD[777] = 2.57507000E+00;
    COFD[778] = -1.24033737E-01;
    COFD[779] = 5.56694959E-03;
    COFD[780] = -1.46550083E+01;
    COFD[781] = 3.83606243E+00;
    COFD[782] = -2.86076532E-01;
    COFD[783] = 1.25205829E-02;
    COFD[784] = -1.43340796E+01;
    COFD[785] = 3.17651319E+00;
    COFD[786] = -2.02028974E-01;
    COFD[787] = 8.94232502E-03;
    COFD[788] = -2.12652533E+01;
    COFD[789] = 5.59961818E+00;
    COFD[790] = -4.91624858E-01;
    COFD[791] = 2.05035550E-02;
    COFD[792] = -1.55511344E+01;
    COFD[793] = 3.48070094E+00;
    COFD[794] = -2.40859499E-01;
    COFD[795] = 1.05972514E-02;
    COFD[796] = -1.55588279E+01;
    COFD[797] = 3.48070094E+00;
    COFD[798] = -2.40859499E-01;
    COFD[799] = 1.05972514E-02;
    COFD[800] = -1.55661750E+01;
    COFD[801] = 3.48070094E+00;
    COFD[802] = -2.40859499E-01;
    COFD[803] = 1.05972514E-02;
    COFD[804] = -1.84688406E+01;
    COFD[805] = 4.49330851E+00;
    COFD[806] = -3.68208715E-01;
    COFD[807] = 1.59565402E-02;
    COFD[808] = -1.52721107E+01;
    COFD[809] = 3.36790500E+00;
    COFD[810] = -2.26321740E-01;
    COFD[811] = 9.97135055E-03;
    COFD[812] = -2.08293255E+01;
    COFD[813] = 5.35267674E+00;
    COFD[814] = -4.69010505E-01;
    COFD[815] = 1.98979152E-02;
    COFD[816] = -1.63254691E+01;
    COFD[817] = 3.82388595E+00;
    COFD[818] = -2.84480724E-01;
    COFD[819] = 1.24506311E-02;
    COFD[820] = -1.63493345E+01;
    COFD[821] = 3.82388595E+00;
    COFD[822] = -2.84480724E-01;
    COFD[823] = 1.24506311E-02;
    COFD[824] = -1.62724462E+01;
    COFD[825] = 3.79163564E+00;
    COFD[826] = -2.80257365E-01;
    COFD[827] = 1.22656902E-02;
    COFD[828] = -2.07595845E+01;
    COFD[829] = 5.32244593E+00;
    COFD[830] = -4.65829403E-01;
    COFD[831] = 1.97895274E-02;
    COFD[832] = -1.85844688E+01;
    COFD[833] = 4.51052425E+00;
    COFD[834] = -3.70301627E-01;
    COFD[835] = 1.60416153E-02;
    COFD[836] = -2.05184870E+01;
    COFD[837] = 5.18417470E+00;
    COFD[838] = -4.49491573E-01;
    COFD[839] = 1.91438508E-02;
    COFD[840] = -2.05375724E+01;
    COFD[841] = 5.18417470E+00;
    COFD[842] = -4.49491573E-01;
    COFD[843] = 1.91438508E-02;
    COFD[844] = -2.08463209E+01;
    COFD[845] = 5.32244593E+00;
    COFD[846] = -4.65829403E-01;
    COFD[847] = 1.97895274E-02;
    COFD[848] = -1.86507213E+01;
    COFD[849] = 4.60874797E+00;
    COFD[850] = -3.82368716E-01;
    COFD[851] = 1.65370164E-02;
    COFD[852] = -1.93917298E+01;
    COFD[853] = 4.78708023E+00;
    COFD[854] = -4.03693144E-01;
    COFD[855] = 1.73884817E-02;
    COFD[856] = -1.82145353E+01;
    COFD[857] = 4.46848269E+00;
    COFD[858] = -3.65269718E-01;
    COFD[859] = 1.58407652E-02;
    COFD[860] = -2.05235731E+01;
    COFD[861] = 5.18417470E+00;
    COFD[862] = -4.49491573E-01;
    COFD[863] = 1.91438508E-02;
    COFD[864] = -1.95875976E+01;
    COFD[865] = 4.84393038E+00;
    COFD[866] = -4.10274737E-01;
    COFD[867] = 1.76417458E-02;
    COFD[868] = -1.94912151E+01;
    COFD[869] = 4.81575071E+00;
    COFD[870] = -4.07042139E-01;
    COFD[871] = 1.75187504E-02;
    COFD[872] = -2.13698722E+01;
    COFD[873] = 5.34971865E+00;
    COFD[874] = -4.68771123E-01;
    COFD[875] = 1.98933811E-02;
    COFD[876] = -2.01315602E+01;
    COFD[877] = 4.97613338E+00;
    COFD[878] = -4.26175206E-01;
    COFD[879] = 1.82809270E-02;
    COFD[880] = -2.00964665E+01;
    COFD[881] = 4.96870443E+00;
    COFD[882] = -4.25292447E-01;
    COFD[883] = 1.82459096E-02;
    COFD[884] = -2.00997774E+01;
    COFD[885] = 4.96870443E+00;
    COFD[886] = -4.25292447E-01;
    COFD[887] = 1.82459096E-02;
    COFD[888] = -2.13955999E+01;
    COFD[889] = 5.25183817E+00;
    COFD[890] = -4.57376333E-01;
    COFD[891] = 1.94504429E-02;
    COFD[892] = -2.13968281E+01;
    COFD[893] = 5.25183817E+00;
    COFD[894] = -4.57376333E-01;
    COFD[895] = 1.94504429E-02;
    COFD[896] = -1.52486273E+01;
    COFD[897] = 3.35922578E+00;
    COFD[898] = -2.25181399E-01;
    COFD[899] = 9.92132878E-03;
    COFD[900] = -1.43190389E+01;
    COFD[901] = 3.17651319E+00;
    COFD[902] = -2.02028974E-01;
    COFD[903] = 8.94232502E-03;
    COFD[904] = -1.18998012E+01;
    COFD[905] = 2.57507000E+00;
    COFD[906] = -1.24033737E-01;
    COFD[907] = 5.56694959E-03;
    COFD[908] = -1.46554748E+01;
    COFD[909] = 3.83606243E+00;
    COFD[910] = -2.86076532E-01;
    COFD[911] = 1.25205829E-02;
    COFD[912] = -1.43394069E+01;
    COFD[913] = 3.17651319E+00;
    COFD[914] = -2.02028974E-01;
    COFD[915] = 8.94232502E-03;
    COFD[916] = -2.06463744E+01;
    COFD[917] = 5.41688482E+00;
    COFD[918] = -4.73387188E-01;
    COFD[919] = 1.99280175E-02;
    COFD[920] = -1.55588279E+01;
    COFD[921] = 3.48070094E+00;
    COFD[922] = -2.40859499E-01;
    COFD[923] = 1.05972514E-02;
    COFD[924] = -1.55666415E+01;
    COFD[925] = 3.48070094E+00;
    COFD[926] = -2.40859499E-01;
    COFD[927] = 1.05972514E-02;
    COFD[928] = -1.55741053E+01;
    COFD[929] = 3.48070094E+00;
    COFD[930] = -2.40859499E-01;
    COFD[931] = 1.05972514E-02;
    COFD[932] = -1.84777607E+01;
    COFD[933] = 4.49330851E+00;
    COFD[934] = -3.68208715E-01;
    COFD[935] = 1.59565402E-02;
    COFD[936] = -1.52792891E+01;
    COFD[937] = 3.36790500E+00;
    COFD[938] = -2.26321740E-01;
    COFD[939] = 9.97135055E-03;
    COFD[940] = -2.08367725E+01;
    COFD[941] = 5.35267674E+00;
    COFD[942] = -4.69010505E-01;
    COFD[943] = 1.98979152E-02;
    COFD[944] = -1.63301444E+01;
    COFD[945] = 3.82388595E+00;
    COFD[946] = -2.84480724E-01;
    COFD[947] = 1.24506311E-02;
    COFD[948] = -1.63542394E+01;
    COFD[949] = 3.82388595E+00;
    COFD[950] = -2.84480724E-01;
    COFD[951] = 1.24506311E-02;
    COFD[952] = -1.62775714E+01;
    COFD[953] = 3.79163564E+00;
    COFD[954] = -2.80257365E-01;
    COFD[955] = 1.22656902E-02;
    COFD[956] = -2.07672833E+01;
    COFD[957] = 5.32244593E+00;
    COFD[958] = -4.65829403E-01;
    COFD[959] = 1.97895274E-02;
    COFD[960] = -1.85919214E+01;
    COFD[961] = 4.51052425E+00;
    COFD[962] = -3.70301627E-01;
    COFD[963] = 1.60416153E-02;
    COFD[964] = -2.05272328E+01;
    COFD[965] = 5.18417470E+00;
    COFD[966] = -4.49491573E-01;
    COFD[967] = 1.91438508E-02;
    COFD[968] = -2.05466616E+01;
    COFD[969] = 5.18417470E+00;
    COFD[970] = -4.49491573E-01;
    COFD[971] = 1.91438508E-02;
    COFD[972] = -2.08554914E+01;
    COFD[973] = 5.32244593E+00;
    COFD[974] = -4.65829403E-01;
    COFD[975] = 1.97895274E-02;
    COFD[976] = -1.86576191E+01;
    COFD[977] = 4.60874797E+00;
    COFD[978] = -3.82368716E-01;
    COFD[979] = 1.65370164E-02;
    COFD[980] = -1.94004795E+01;
    COFD[981] = 4.78708023E+00;
    COFD[982] = -4.03693144E-01;
    COFD[983] = 1.73884817E-02;
    COFD[984] = -1.82217198E+01;
    COFD[985] = 4.46848269E+00;
    COFD[986] = -3.65269718E-01;
    COFD[987] = 1.58407652E-02;
    COFD[988] = -2.05324091E+01;
    COFD[989] = 5.18417470E+00;
    COFD[990] = -4.49491573E-01;
    COFD[991] = 1.91438508E-02;
    COFD[992] = -1.95961596E+01;
    COFD[993] = 4.84393038E+00;
    COFD[994] = -4.10274737E-01;
    COFD[995] = 1.76417458E-02;
    COFD[996] = -1.94998722E+01;
    COFD[997] = 4.81575071E+00;
    COFD[998] = -4.07042139E-01;
    COFD[999] = 1.75187504E-02;
    COFD[1000] = -2.12907159E+01;
    COFD[1001] = 5.32167660E+00;
    COFD[1002] = -4.65740624E-01;
    COFD[1003] = 1.97861081E-02;
    COFD[1004] = -2.01412473E+01;
    COFD[1005] = 4.97613338E+00;
    COFD[1006] = -4.26175206E-01;
    COFD[1007] = 1.82809270E-02;
    COFD[1008] = -2.01062206E+01;
    COFD[1009] = 4.96870443E+00;
    COFD[1010] = -4.25292447E-01;
    COFD[1011] = 1.82459096E-02;
    COFD[1012] = -2.01095969E+01;
    COFD[1013] = 4.96870443E+00;
    COFD[1014] = -4.25292447E-01;
    COFD[1015] = 1.82459096E-02;
    COFD[1016] = -2.14072803E+01;
    COFD[1017] = 5.25183817E+00;
    COFD[1018] = -4.57376333E-01;
    COFD[1019] = 1.94504429E-02;
    COFD[1020] = -2.14085375E+01;
    COFD[1021] = 5.25183817E+00;
    COFD[1022] = -4.57376333E-01;
    COFD[1023] = 1.94504429E-02;
    COFD[1024] = -1.52554761E+01;
    COFD[1025] = 3.35922578E+00;
    COFD[1026] = -2.25181399E-01;
    COFD[1027] = 9.92132878E-03;
    COFD[1028] = -1.43238998E+01;
    COFD[1029] = 3.17651319E+00;
    COFD[1030] = -2.02028974E-01;
    COFD[1031] = 8.94232502E-03;
    COFD[1032] = -1.19006548E+01;
    COFD[1033] = 2.57507000E+00;
    COFD[1034] = -1.24033737E-01;
    COFD[1035] = 5.56694959E-03;
    COFD[1036] = -1.46559141E+01;
    COFD[1037] = 3.83606243E+00;
    COFD[1038] = -2.86076532E-01;
    COFD[1039] = 1.25205829E-02;
    COFD[1040] = -1.43444709E+01;
    COFD[1041] = 3.17651319E+00;
    COFD[1042] = -2.02028974E-01;
    COFD[1043] = 8.94232502E-03;
    COFD[1044] = -2.06516336E+01;
    COFD[1045] = 5.41688482E+00;
    COFD[1046] = -4.73387188E-01;
    COFD[1047] = 1.99280175E-02;
    COFD[1048] = -1.55661750E+01;
    COFD[1049] = 3.48070094E+00;
    COFD[1050] = -2.40859499E-01;
    COFD[1051] = 1.05972514E-02;
    COFD[1052] = -1.55741053E+01;
    COFD[1053] = 3.48070094E+00;
    COFD[1054] = -2.40859499E-01;
    COFD[1055] = 1.05972514E-02;
    COFD[1056] = -1.55816822E+01;
    COFD[1057] = 3.48070094E+00;
    COFD[1058] = -2.40859499E-01;
    COFD[1059] = 1.05972514E-02;
    COFD[1060] = -1.84863000E+01;
    COFD[1061] = 4.49330851E+00;
    COFD[1062] = -3.68208715E-01;
    COFD[1063] = 1.59565402E-02;
    COFD[1064] = -1.52861376E+01;
    COFD[1065] = 3.36790500E+00;
    COFD[1066] = -2.26321740E-01;
    COFD[1067] = 9.97135055E-03;
    COFD[1068] = -2.08438809E+01;
    COFD[1069] = 5.35267674E+00;
    COFD[1070] = -4.69010505E-01;
    COFD[1071] = 1.98979152E-02;
    COFD[1072] = -1.63345829E+01;
    COFD[1073] = 3.82388595E+00;
    COFD[1074] = -2.84480724E-01;
    COFD[1075] = 1.24506311E-02;
    COFD[1076] = -1.63588981E+01;
    COFD[1077] = 3.82388595E+00;
    COFD[1078] = -2.84480724E-01;
    COFD[1079] = 1.24506311E-02;
    COFD[1080] = -1.62824412E+01;
    COFD[1081] = 3.79163564E+00;
    COFD[1082] = -2.80257365E-01;
    COFD[1083] = 1.22656902E-02;
    COFD[1084] = -2.07746356E+01;
    COFD[1085] = 5.32244593E+00;
    COFD[1086] = -4.65829403E-01;
    COFD[1087] = 1.97895274E-02;
    COFD[1088] = -1.85990352E+01;
    COFD[1089] = 4.51052425E+00;
    COFD[1090] = -3.70301627E-01;
    COFD[1091] = 1.60416153E-02;
    COFD[1092] = -2.05356023E+01;
    COFD[1093] = 5.18417470E+00;
    COFD[1094] = -4.49491573E-01;
    COFD[1095] = 1.91438508E-02;
    COFD[1096] = -2.05553656E+01;
    COFD[1097] = 5.18417470E+00;
    COFD[1098] = -4.49491573E-01;
    COFD[1099] = 1.91438508E-02;
    COFD[1100] = -2.08642748E+01;
    COFD[1101] = 5.32244593E+00;
    COFD[1102] = -4.65829403E-01;
    COFD[1103] = 1.97895274E-02;
    COFD[1104] = -1.86641962E+01;
    COFD[1105] = 4.60874797E+00;
    COFD[1106] = -3.82368716E-01;
    COFD[1107] = 1.65370164E-02;
    COFD[1108] = -1.94088529E+01;
    COFD[1109] = 4.78708023E+00;
    COFD[1110] = -4.03693144E-01;
    COFD[1111] = 1.73884817E-02;
    COFD[1112] = -1.82285740E+01;
    COFD[1113] = 4.46848269E+00;
    COFD[1114] = -3.65269718E-01;
    COFD[1115] = 1.58407652E-02;
    COFD[1116] = -2.05408665E+01;
    COFD[1117] = 5.18417470E+00;
    COFD[1118] = -4.49491573E-01;
    COFD[1119] = 1.91438508E-02;
    COFD[1120] = -1.96043503E+01;
    COFD[1121] = 4.84393038E+00;
    COFD[1122] = -4.10274737E-01;
    COFD[1123] = 1.76417458E-02;
    COFD[1124] = -1.95081555E+01;
    COFD[1125] = 4.81575071E+00;
    COFD[1126] = -4.07042139E-01;
    COFD[1127] = 1.75187504E-02;
    COFD[1128] = -2.13011157E+01;
    COFD[1129] = 5.32167660E+00;
    COFD[1130] = -4.65740624E-01;
    COFD[1131] = 1.97861081E-02;
    COFD[1132] = -2.01505348E+01;
    COFD[1133] = 4.97613338E+00;
    COFD[1134] = -4.26175206E-01;
    COFD[1135] = 1.82809270E-02;
    COFD[1136] = -2.01155735E+01;
    COFD[1137] = 4.96870443E+00;
    COFD[1138] = -4.25292447E-01;
    COFD[1139] = 1.82459096E-02;
    COFD[1140] = -2.01190139E+01;
    COFD[1141] = 4.96870443E+00;
    COFD[1142] = -4.25292447E-01;
    COFD[1143] = 1.82459096E-02;
    COFD[1144] = -2.14185232E+01;
    COFD[1145] = 5.25183817E+00;
    COFD[1146] = -4.57376333E-01;
    COFD[1147] = 1.94504429E-02;
    COFD[1148] = -2.14198091E+01;
    COFD[1149] = 5.25183817E+00;
    COFD[1150] = -4.57376333E-01;
    COFD[1151] = 1.94504429E-02;
    COFD[1152] = -1.81432461E+01;
    COFD[1153] = 4.37565431E+00;
    COFD[1154] = -3.53906025E-01;
    COFD[1155] = 1.53760786E-02;
    COFD[1156] = -1.70534856E+01;
    COFD[1157] = 4.14240922E+00;
    COFD[1158] = -3.25239774E-01;
    COFD[1159] = 1.41980687E-02;
    COFD[1160] = -1.37794315E+01;
    COFD[1161] = 3.23973858E+00;
    COFD[1162] = -2.09989036E-01;
    COFD[1163] = 9.27667906E-03;
    COFD[1164] = -1.76147026E+01;
    COFD[1165] = 4.86049500E+00;
    COFD[1166] = -4.12200578E-01;
    COFD[1167] = 1.77160971E-02;
    COFD[1168] = -1.70757047E+01;
    COFD[1169] = 4.14240922E+00;
    COFD[1170] = -3.25239774E-01;
    COFD[1171] = 1.41980687E-02;
    COFD[1172] = -2.07653719E+01;
    COFD[1173] = 5.01092022E+00;
    COFD[1174] = -3.77985635E-01;
    COFD[1175] = 1.40968645E-02;
    COFD[1176] = -1.84688406E+01;
    COFD[1177] = 4.49330851E+00;
    COFD[1178] = -3.68208715E-01;
    COFD[1179] = 1.59565402E-02;
    COFD[1180] = -1.84777607E+01;
    COFD[1181] = 4.49330851E+00;
    COFD[1182] = -3.68208715E-01;
    COFD[1183] = 1.59565402E-02;
    COFD[1184] = -1.84863000E+01;
    COFD[1185] = 4.49330851E+00;
    COFD[1186] = -3.68208715E-01;
    COFD[1187] = 1.59565402E-02;
    COFD[1188] = -2.13425698E+01;
    COFD[1189] = 5.40460130E+00;
    COFD[1190] = -4.72718910E-01;
    COFD[1191] = 1.99362717E-02;
    COFD[1192] = -1.81735763E+01;
    COFD[1193] = 4.38391495E+00;
    COFD[1194] = -3.54941287E-01;
    COFD[1195] = 1.54195107E-02;
    COFD[1196] = -2.19317743E+01;
    COFD[1197] = 5.45216133E+00;
    COFD[1198] = -4.52916925E-01;
    COFD[1199] = 1.80456400E-02;
    COFD[1200] = -1.93015555E+01;
    COFD[1201] = 4.85015581E+00;
    COFD[1202] = -4.10945109E-01;
    COFD[1203] = 1.76651398E-02;
    COFD[1204] = -1.93276434E+01;
    COFD[1205] = 4.85015581E+00;
    COFD[1206] = -4.10945109E-01;
    COFD[1207] = 1.76651398E-02;
    COFD[1208] = -1.92867554E+01;
    COFD[1209] = 4.83375900E+00;
    COFD[1210] = -4.09146560E-01;
    COFD[1211] = 1.76006599E-02;
    COFD[1212] = -2.20063594E+01;
    COFD[1213] = 5.48540187E+00;
    COFD[1214] = -4.58962148E-01;
    COFD[1215] = 1.83770355E-02;
    COFD[1216] = -2.14151520E+01;
    COFD[1217] = 5.41122754E+00;
    COFD[1218] = -4.73185889E-01;
    COFD[1219] = 1.99407905E-02;
    COFD[1220] = -2.22116706E+01;
    COFD[1221] = 5.54251230E+00;
    COFD[1222] = -4.70946314E-01;
    COFD[1223] = 1.90785869E-02;
    COFD[1224] = -2.22343363E+01;
    COFD[1225] = 5.54251230E+00;
    COFD[1226] = -4.70946314E-01;
    COFD[1227] = 1.90785869E-02;
    COFD[1228] = -2.21083035E+01;
    COFD[1229] = 5.48540187E+00;
    COFD[1230] = -4.58962148E-01;
    COFD[1231] = 1.83770355E-02;
    COFD[1232] = -2.13961414E+01;
    COFD[1233] = 5.46685775E+00;
    COFD[1234] = -4.78665416E-01;
    COFD[1235] = 2.01093915E-02;
    COFD[1236] = -2.20725883E+01;
    COFD[1237] = 5.59642965E+00;
    COFD[1238] = -4.91577716E-01;
    COFD[1239] = 2.05159582E-02;
    COFD[1240] = -2.11031143E+01;
    COFD[1241] = 5.39439999E+00;
    COFD[1242] = -4.72050184E-01;
    COFD[1243] = 1.99336257E-02;
    COFD[1244] = -2.22176950E+01;
    COFD[1245] = 5.54251230E+00;
    COFD[1246] = -4.70946314E-01;
    COFD[1247] = 1.90785869E-02;
    COFD[1248] = -2.21697404E+01;
    COFD[1249] = 5.60807471E+00;
    COFD[1250] = -4.91339309E-01;
    COFD[1251] = 2.04365761E-02;
    COFD[1252] = -2.21216828E+01;
    COFD[1253] = 5.60203389E+00;
    COFD[1254] = -4.91444416E-01;
    COFD[1255] = 2.04761886E-02;
    COFD[1256] = -2.25168081E+01;
    COFD[1257] = 5.46125558E+00;
    COFD[1258] = -4.54580949E-01;
    COFD[1259] = 1.81370928E-02;
    COFD[1260] = -2.23890317E+01;
    COFD[1261] = 5.59178974E+00;
    COFD[1262] = -4.85668031E-01;
    COFD[1263] = 2.00491907E-02;
    COFD[1264] = -2.23772680E+01;
    COFD[1265] = 5.59425354E+00;
    COFD[1266] = -4.86232980E-01;
    COFD[1267] = 2.00835981E-02;
    COFD[1268] = -2.23812726E+01;
    COFD[1269] = 5.59425354E+00;
    COFD[1270] = -4.86232980E-01;
    COFD[1271] = 2.00835981E-02;
    COFD[1272] = -2.28655752E+01;
    COFD[1273] = 5.50522401E+00;
    COFD[1274] = -4.63604304E-01;
    COFD[1275] = 1.86600785E-02;
    COFD[1276] = -2.28671232E+01;
    COFD[1277] = 5.50522401E+00;
    COFD[1278] = -4.63604304E-01;
    COFD[1279] = 1.86600785E-02;
    COFD[1280] = -1.50031687E+01;
    COFD[1281] = 3.26223357E+00;
    COFD[1282] = -2.12746642E-01;
    COFD[1283] = 9.38912883E-03;
    COFD[1284] = -1.40999008E+01;
    COFD[1285] = 3.08120012E+00;
    COFD[1286] = -1.89629903E-01;
    COFD[1287] = 8.40361952E-03;
    COFD[1288] = -1.17159737E+01;
    COFD[1289] = 2.48123210E+00;
    COFD[1290] = -1.11322604E-01;
    COFD[1291] = 4.99282389E-03;
    COFD[1292] = -1.43151174E+01;
    COFD[1293] = 3.68038508E+00;
    COFD[1294] = -2.65779346E-01;
    COFD[1295] = 1.16360771E-02;
    COFD[1296] = -1.41191261E+01;
    COFD[1297] = 3.08120012E+00;
    COFD[1298] = -1.89629903E-01;
    COFD[1299] = 8.40361952E-03;
    COFD[1300] = -2.11388331E+01;
    COFD[1301] = 5.55529675E+00;
    COFD[1302] = -4.87942518E-01;
    COFD[1303] = 2.04249054E-02;
    COFD[1304] = -1.52721107E+01;
    COFD[1305] = 3.36790500E+00;
    COFD[1306] = -2.26321740E-01;
    COFD[1307] = 9.97135055E-03;
    COFD[1308] = -1.52792891E+01;
    COFD[1309] = 3.36790500E+00;
    COFD[1310] = -2.26321740E-01;
    COFD[1311] = 9.97135055E-03;
    COFD[1312] = -1.52861376E+01;
    COFD[1313] = 3.36790500E+00;
    COFD[1314] = -2.26321740E-01;
    COFD[1315] = 9.97135055E-03;
    COFD[1316] = -1.81735763E+01;
    COFD[1317] = 4.38391495E+00;
    COFD[1318] = -3.54941287E-01;
    COFD[1319] = 1.54195107E-02;
    COFD[1320] = -1.50233475E+01;
    COFD[1321] = 3.26660767E+00;
    COFD[1322] = -2.13287177E-01;
    COFD[1323] = 9.41137857E-03;
    COFD[1324] = -2.05128705E+01;
    COFD[1325] = 5.23843909E+00;
    COFD[1326] = -4.55815614E-01;
    COFD[1327] = 1.93898040E-02;
    COFD[1328] = -1.59634533E+01;
    COFD[1329] = 3.67388294E+00;
    COFD[1330] = -2.64990709E-01;
    COFD[1331] = 1.16042706E-02;
    COFD[1332] = -1.59863030E+01;
    COFD[1333] = 3.67388294E+00;
    COFD[1334] = -2.64990709E-01;
    COFD[1335] = 1.16042706E-02;
    COFD[1336] = -1.59525102E+01;
    COFD[1337] = 3.66023858E+00;
    COFD[1338] = -2.63401043E-01;
    COFD[1339] = 1.15432000E-02;
    COFD[1340] = -2.04144604E+01;
    COFD[1341] = 5.19614628E+00;
    COFD[1342] = -4.50889164E-01;
    COFD[1343] = 1.91983328E-02;
    COFD[1344] = -1.82955252E+01;
    COFD[1345] = 4.40289649E+00;
    COFD[1346] = -3.57289765E-01;
    COFD[1347] = 1.55166804E-02;
    COFD[1348] = -2.02922701E+01;
    COFD[1349] = 5.11106992E+00;
    COFD[1350] = -4.42047129E-01;
    COFD[1351] = 1.89042990E-02;
    COFD[1352] = -2.03099025E+01;
    COFD[1353] = 5.11106992E+00;
    COFD[1354] = -4.42047129E-01;
    COFD[1355] = 1.89042990E-02;
    COFD[1356] = -2.04949373E+01;
    COFD[1357] = 5.19614628E+00;
    COFD[1358] = -4.50889164E-01;
    COFD[1359] = 1.91983328E-02;
    COFD[1360] = -1.83296965E+01;
    COFD[1361] = 4.48570999E+00;
    COFD[1362] = -3.67301524E-01;
    COFD[1363] = 1.59204254E-02;
    COFD[1364] = -1.91118445E+01;
    COFD[1365] = 4.68715685E+00;
    COFD[1366] = -3.91979493E-01;
    COFD[1367] = 1.69314004E-02;
    COFD[1368] = -1.79116531E+01;
    COFD[1369] = 4.35148286E+00;
    COFD[1370] = -3.50886647E-01;
    COFD[1371] = 1.52498573E-02;
    COFD[1372] = -2.02969740E+01;
    COFD[1373] = 5.11106992E+00;
    COFD[1374] = -4.42047129E-01;
    COFD[1375] = 1.89042990E-02;
    COFD[1376] = -1.93064215E+01;
    COFD[1377] = 4.74387793E+00;
    COFD[1378] = -3.98574972E-01;
    COFD[1379] = 1.71862289E-02;
    COFD[1380] = -1.92044492E+01;
    COFD[1381] = 4.71304783E+00;
    COFD[1382] = -3.94942083E-01;
    COFD[1383] = 1.70435959E-02;
    COFD[1384] = -2.10310742E+01;
    COFD[1385] = 5.23485505E+00;
    COFD[1386] = -4.55400362E-01;
    COFD[1387] = 1.93737680E-02;
    COFD[1388] = -1.97709603E+01;
    COFD[1389] = 4.84731557E+00;
    COFD[1390] = -4.10638352E-01;
    COFD[1391] = 1.76543886E-02;
    COFD[1392] = -1.97422209E+01;
    COFD[1393] = 4.84249900E+00;
    COFD[1394] = -4.10120448E-01;
    COFD[1395] = 1.76363500E-02;
    COFD[1396] = -1.97452574E+01;
    COFD[1397] = 4.84249900E+00;
    COFD[1398] = -4.10120448E-01;
    COFD[1399] = 1.76363500E-02;
    COFD[1400] = -2.10844012E+01;
    COFD[1401] = 5.15315713E+00;
    COFD[1402] = -4.46344043E-01;
    COFD[1403] = 1.90431546E-02;
    COFD[1404] = -2.10855099E+01;
    COFD[1405] = 5.15315713E+00;
    COFD[1406] = -4.46344043E-01;
    COFD[1407] = 1.90431546E-02;
    COFD[1408] = -2.04833713E+01;
    COFD[1409] = 5.23112374E+00;
    COFD[1410] = -4.54967682E-01;
    COFD[1411] = 1.93570423E-02;
    COFD[1412] = -1.94373127E+01;
    COFD[1413] = 5.02567894E+00;
    COFD[1414] = -4.32045169E-01;
    COFD[1415] = 1.85132214E-02;
    COFD[1416] = -1.60528285E+01;
    COFD[1417] = 4.11188603E+00;
    COFD[1418] = -3.21540884E-01;
    COFD[1419] = 1.40482564E-02;
    COFD[1420] = -1.97550088E+01;
    COFD[1421] = 5.56931926E+00;
    COFD[1422] = -4.89105511E-01;
    COFD[1423] = 2.04493129E-02;
    COFD[1424] = -1.94570287E+01;
    COFD[1425] = 5.02567894E+00;
    COFD[1426] = -4.32045169E-01;
    COFD[1427] = 1.85132214E-02;
    COFD[1428] = -1.77563250E+01;
    COFD[1429] = 3.57475686E+00;
    COFD[1430] = -1.56396297E-01;
    COFD[1431] = 3.12157721E-03;
    COFD[1432] = -2.08293255E+01;
    COFD[1433] = 5.35267674E+00;
    COFD[1434] = -4.69010505E-01;
    COFD[1435] = 1.98979152E-02;
    COFD[1436] = -2.08367725E+01;
    COFD[1437] = 5.35267674E+00;
    COFD[1438] = -4.69010505E-01;
    COFD[1439] = 1.98979152E-02;
    COFD[1440] = -2.08438809E+01;
    COFD[1441] = 5.35267674E+00;
    COFD[1442] = -4.69010505E-01;
    COFD[1443] = 1.98979152E-02;
    COFD[1444] = -2.19317743E+01;
    COFD[1445] = 5.45216133E+00;
    COFD[1446] = -4.52916925E-01;
    COFD[1447] = 1.80456400E-02;
    COFD[1448] = -2.05128705E+01;
    COFD[1449] = 5.23843909E+00;
    COFD[1450] = -4.55815614E-01;
    COFD[1451] = 1.93898040E-02;
    COFD[1452] = -1.90499441E+01;
    COFD[1453] = 3.99221757E+00;
    COFD[1454] = -2.19854880E-01;
    COFD[1455] = 6.22736279E-03;
    COFD[1456] = -2.14215700E+01;
    COFD[1457] = 5.56531152E+00;
    COFD[1458] = -4.88789821E-01;
    COFD[1459] = 2.04437116E-02;
    COFD[1460] = -2.14449559E+01;
    COFD[1461] = 5.56531152E+00;
    COFD[1462] = -4.88789821E-01;
    COFD[1463] = 2.04437116E-02;
    COFD[1464] = -2.14082453E+01;
    COFD[1465] = 5.55346617E+00;
    COFD[1466] = -4.87783156E-01;
    COFD[1467] = 2.04210886E-02;
    COFD[1468] = -1.93214527E+01;
    COFD[1469] = 4.10954793E+00;
    COFD[1470] = -2.37523329E-01;
    COFD[1471] = 7.08858141E-03;
    COFD[1472] = -2.19786173E+01;
    COFD[1473] = 5.43750833E+00;
    COFD[1474] = -4.50273329E-01;
    COFD[1475] = 1.79013718E-02;
    COFD[1476] = -2.01015340E+01;
    COFD[1477] = 4.41511629E+00;
    COFD[1478] = -2.84086963E-01;
    COFD[1479] = 9.37586971E-03;
    COFD[1480] = -2.01199204E+01;
    COFD[1481] = 4.41511629E+00;
    COFD[1482] = -2.84086963E-01;
    COFD[1483] = 9.37586971E-03;
    COFD[1484] = -1.94051843E+01;
    COFD[1485] = 4.10954793E+00;
    COFD[1486] = -2.37523329E-01;
    COFD[1487] = 7.08858141E-03;
    COFD[1488] = -2.16798265E+01;
    COFD[1489] = 5.36811769E+00;
    COFD[1490] = -4.37727086E-01;
    COFD[1491] = 1.72167686E-02;
    COFD[1492] = -2.15802788E+01;
    COFD[1493] = 5.16868516E+00;
    COFD[1494] = -4.03721581E-01;
    COFD[1495] = 1.54206640E-02;
    COFD[1496] = -2.17855148E+01;
    COFD[1497] = 5.47519298E+00;
    COFD[1498] = -4.57113040E-01;
    COFD[1499] = 1.82758312E-02;
    COFD[1500] = -2.01064363E+01;
    COFD[1501] = 4.41511629E+00;
    COFD[1502] = -2.84086963E-01;
    COFD[1503] = 9.37586971E-03;
    COFD[1504] = -2.14453157E+01;
    COFD[1505] = 5.07680397E+00;
    COFD[1506] = -3.88612087E-01;
    COFD[1507] = 1.46395101E-02;
    COFD[1508] = -2.15258568E+01;
    COFD[1509] = 5.12799307E+00;
    COFD[1510] = -3.96938732E-01;
    COFD[1511] = 1.50673195E-02;
    COFD[1512] = -1.98359760E+01;
    COFD[1513] = 4.11158627E+00;
    COFD[1514] = -2.37831519E-01;
    COFD[1515] = 7.10363413E-03;
    COFD[1516] = -2.12219677E+01;
    COFD[1517] = 4.87252053E+00;
    COFD[1518] = -3.56127804E-01;
    COFD[1519] = 1.29948788E-02;
    COFD[1520] = -2.12330900E+01;
    COFD[1521] = 4.88535789E+00;
    COFD[1522] = -3.58153894E-01;
    COFD[1523] = 1.30969624E-02;
    COFD[1524] = -2.12362684E+01;
    COFD[1525] = 4.88535789E+00;
    COFD[1526] = -3.58153894E-01;
    COFD[1527] = 1.30969624E-02;
    COFD[1528] = -2.04620510E+01;
    COFD[1529] = 4.26473557E+00;
    COFD[1530] = -2.61033037E-01;
    COFD[1531] = 8.23906412E-03;
    COFD[1532] = -2.04632210E+01;
    COFD[1533] = 4.26473557E+00;
    COFD[1534] = -2.61033037E-01;
    COFD[1535] = 8.23906412E-03;
    COFD[1536] = -1.59404882E+01;
    COFD[1537] = 3.66853818E+00;
    COFD[1538] = -2.64346221E-01;
    COFD[1539] = 1.15784613E-02;
    COFD[1540] = -1.50584249E+01;
    COFD[1541] = 3.47945612E+00;
    COFD[1542] = -2.40703722E-01;
    COFD[1543] = 1.05907441E-02;
    COFD[1544] = -1.25098960E+01;
    COFD[1545] = 2.77873601E+00;
    COFD[1546] = -1.50637360E-01;
    COFD[1547] = 6.72684281E-03;
    COFD[1548] = -1.57972369E+01;
    COFD[1549] = 4.22225052E+00;
    COFD[1550] = -3.35156428E-01;
    COFD[1551] = 1.46104855E-02;
    COFD[1552] = -1.50724636E+01;
    COFD[1553] = 3.47945612E+00;
    COFD[1554] = -2.40703722E-01;
    COFD[1555] = 1.05907441E-02;
    COFD[1556] = -2.12639214E+01;
    COFD[1557] = 5.61184117E+00;
    COFD[1558] = -4.90532156E-01;
    COFD[1559] = 2.03507922E-02;
    COFD[1560] = -1.63254691E+01;
    COFD[1561] = 3.82388595E+00;
    COFD[1562] = -2.84480724E-01;
    COFD[1563] = 1.24506311E-02;
    COFD[1564] = -1.63301444E+01;
    COFD[1565] = 3.82388595E+00;
    COFD[1566] = -2.84480724E-01;
    COFD[1567] = 1.24506311E-02;
    COFD[1568] = -1.63345829E+01;
    COFD[1569] = 3.82388595E+00;
    COFD[1570] = -2.84480724E-01;
    COFD[1571] = 1.24506311E-02;
    COFD[1572] = -1.93015555E+01;
    COFD[1573] = 4.85015581E+00;
    COFD[1574] = -4.10945109E-01;
    COFD[1575] = 1.76651398E-02;
    COFD[1576] = -1.59634533E+01;
    COFD[1577] = 3.67388294E+00;
    COFD[1578] = -2.64990709E-01;
    COFD[1579] = 1.16042706E-02;
    COFD[1580] = -2.14215700E+01;
    COFD[1581] = 5.56531152E+00;
    COFD[1582] = -4.88789821E-01;
    COFD[1583] = 2.04437116E-02;
    COFD[1584] = -1.73027557E+01;
    COFD[1585] = 4.21416723E+00;
    COFD[1586] = -3.34163932E-01;
    COFD[1587] = 1.45697432E-02;
    COFD[1588] = -1.73198034E+01;
    COFD[1589] = 4.21416723E+00;
    COFD[1590] = -3.34163932E-01;
    COFD[1591] = 1.45697432E-02;
    COFD[1592] = -1.72556729E+01;
    COFD[1593] = 4.19029808E+00;
    COFD[1594] = -3.31177076E-01;
    COFD[1595] = 1.44446234E-02;
    COFD[1596] = -2.13538553E+01;
    COFD[1597] = 5.54007827E+00;
    COFD[1598] = -4.86434511E-01;
    COFD[1599] = 2.03779006E-02;
    COFD[1600] = -1.94585111E+01;
    COFD[1601] = 4.87180830E+00;
    COFD[1602] = -4.13582958E-01;
    COFD[1603] = 1.77726094E-02;
    COFD[1604] = -2.11349086E+01;
    COFD[1605] = 5.42846112E+00;
    COFD[1606] = -4.74321870E-01;
    COFD[1607] = 1.99459749E-02;
    COFD[1608] = -2.11458678E+01;
    COFD[1609] = 5.42846112E+00;
    COFD[1610] = -4.74321870E-01;
    COFD[1611] = 1.99459749E-02;
    COFD[1612] = -2.14048982E+01;
    COFD[1613] = 5.54007827E+00;
    COFD[1614] = -4.86434511E-01;
    COFD[1615] = 2.03779006E-02;
    COFD[1616] = -1.95548230E+01;
    COFD[1617] = 4.97133070E+00;
    COFD[1618] = -4.25604177E-01;
    COFD[1619] = 1.82582594E-02;
    COFD[1620] = -2.02434438E+01;
    COFD[1621] = 5.14418672E+00;
    COFD[1622] = -4.45631004E-01;
    COFD[1623] = 1.90308403E-02;
    COFD[1624] = -1.90996795E+01;
    COFD[1625] = 4.82869066E+00;
    COFD[1626] = -4.08564514E-01;
    COFD[1627] = 1.75784675E-02;
    COFD[1628] = -2.11378465E+01;
    COFD[1629] = 5.42846112E+00;
    COFD[1630] = -4.74321870E-01;
    COFD[1631] = 1.99459749E-02;
    COFD[1632] = -2.04054899E+01;
    COFD[1633] = 5.18271974E+00;
    COFD[1634] = -4.49323627E-01;
    COFD[1635] = 1.91373940E-02;
    COFD[1636] = -2.03111230E+01;
    COFD[1637] = 5.15740122E+00;
    COFD[1638] = -4.46644818E-01;
    COFD[1639] = 1.90459001E-02;
    COFD[1640] = -2.17867314E+01;
    COFD[1641] = 5.53950393E+00;
    COFD[1642] = -4.86376204E-01;
    COFD[1643] = 2.03760106E-02;
    COFD[1644] = -2.09217020E+01;
    COFD[1645] = 5.31360223E+00;
    COFD[1646] = -4.64787000E-01;
    COFD[1647] = 1.97483720E-02;
    COFD[1648] = -2.08833669E+01;
    COFD[1649] = 5.30526648E+00;
    COFD[1650] = -4.63785596E-01;
    COFD[1651] = 1.97079873E-02;
    COFD[1652] = -2.08851929E+01;
    COFD[1653] = 5.30526648E+00;
    COFD[1654] = -4.63785596E-01;
    COFD[1655] = 1.97079873E-02;
    COFD[1656] = -2.19248250E+01;
    COFD[1657] = 5.49350509E+00;
    COFD[1658] = -4.81613405E-01;
    COFD[1659] = 2.02171734E-02;
    COFD[1660] = -2.19254485E+01;
    COFD[1661] = 5.49350509E+00;
    COFD[1662] = -4.81613405E-01;
    COFD[1663] = 2.02171734E-02;
    COFD[1664] = -1.59633387E+01;
    COFD[1665] = 3.66853818E+00;
    COFD[1666] = -2.64346221E-01;
    COFD[1667] = 1.15784613E-02;
    COFD[1668] = -1.50766130E+01;
    COFD[1669] = 3.47945612E+00;
    COFD[1670] = -2.40703722E-01;
    COFD[1671] = 1.05907441E-02;
    COFD[1672] = -1.25141260E+01;
    COFD[1673] = 2.77873601E+00;
    COFD[1674] = -1.50637360E-01;
    COFD[1675] = 6.72684281E-03;
    COFD[1676] = -1.57994893E+01;
    COFD[1677] = 4.22225052E+00;
    COFD[1678] = -3.35156428E-01;
    COFD[1679] = 1.46104855E-02;
    COFD[1680] = -1.50911794E+01;
    COFD[1681] = 3.47945612E+00;
    COFD[1682] = -2.40703722E-01;
    COFD[1683] = 1.05907441E-02;
    COFD[1684] = -2.12831323E+01;
    COFD[1685] = 5.61184117E+00;
    COFD[1686] = -4.90532156E-01;
    COFD[1687] = 2.03507922E-02;
    COFD[1688] = -1.63493345E+01;
    COFD[1689] = 3.82388595E+00;
    COFD[1690] = -2.84480724E-01;
    COFD[1691] = 1.24506311E-02;
    COFD[1692] = -1.63542394E+01;
    COFD[1693] = 3.82388595E+00;
    COFD[1694] = -2.84480724E-01;
    COFD[1695] = 1.24506311E-02;
    COFD[1696] = -1.63588981E+01;
    COFD[1697] = 3.82388595E+00;
    COFD[1698] = -2.84480724E-01;
    COFD[1699] = 1.24506311E-02;
    COFD[1700] = -1.93276434E+01;
    COFD[1701] = 4.85015581E+00;
    COFD[1702] = -4.10945109E-01;
    COFD[1703] = 1.76651398E-02;
    COFD[1704] = -1.59863030E+01;
    COFD[1705] = 3.67388294E+00;
    COFD[1706] = -2.64990709E-01;
    COFD[1707] = 1.16042706E-02;
    COFD[1708] = -2.14449559E+01;
    COFD[1709] = 5.56531152E+00;
    COFD[1710] = -4.88789821E-01;
    COFD[1711] = 2.04437116E-02;
    COFD[1712] = -1.73198034E+01;
    COFD[1713] = 4.21416723E+00;
    COFD[1714] = -3.34163932E-01;
    COFD[1715] = 1.45697432E-02;
    COFD[1716] = -1.73374529E+01;
    COFD[1717] = 4.21416723E+00;
    COFD[1718] = -3.34163932E-01;
    COFD[1719] = 1.45697432E-02;
    COFD[1720] = -1.72738845E+01;
    COFD[1721] = 4.19029808E+00;
    COFD[1722] = -3.31177076E-01;
    COFD[1723] = 1.44446234E-02;
    COFD[1724] = -2.13777308E+01;
    COFD[1725] = 5.54007827E+00;
    COFD[1726] = -4.86434511E-01;
    COFD[1727] = 2.03779006E-02;
    COFD[1728] = -1.94819080E+01;
    COFD[1729] = 4.87180830E+00;
    COFD[1730] = -4.13582958E-01;
    COFD[1731] = 1.77726094E-02;
    COFD[1732] = -2.11606963E+01;
    COFD[1733] = 5.42846112E+00;
    COFD[1734] = -4.74321870E-01;
    COFD[1735] = 1.99459749E-02;
    COFD[1736] = -2.11722423E+01;
    COFD[1737] = 5.42846112E+00;
    COFD[1738] = -4.74321870E-01;
    COFD[1739] = 1.99459749E-02;
    COFD[1740] = -2.14314090E+01;
    COFD[1741] = 5.54007827E+00;
    COFD[1742] = -4.86434511E-01;
    COFD[1743] = 2.03779006E-02;
    COFD[1744] = -1.95770968E+01;
    COFD[1745] = 4.97133070E+00;
    COFD[1746] = -4.25604177E-01;
    COFD[1747] = 1.82582594E-02;
    COFD[1748] = -2.02692384E+01;
    COFD[1749] = 5.14418672E+00;
    COFD[1750] = -4.45631004E-01;
    COFD[1751] = 1.90308403E-02;
    COFD[1752] = -1.91225414E+01;
    COFD[1753] = 4.82869066E+00;
    COFD[1754] = -4.08564514E-01;
    COFD[1755] = 1.75784675E-02;
    COFD[1756] = -2.11637902E+01;
    COFD[1757] = 5.42846112E+00;
    COFD[1758] = -4.74321870E-01;
    COFD[1759] = 1.99459749E-02;
    COFD[1760] = -2.04309557E+01;
    COFD[1761] = 5.18271974E+00;
    COFD[1762] = -4.49323627E-01;
    COFD[1763] = 1.91373940E-02;
    COFD[1764] = -2.03367561E+01;
    COFD[1765] = 5.15740122E+00;
    COFD[1766] = -4.46644818E-01;
    COFD[1767] = 1.90459001E-02;
    COFD[1768] = -2.18158049E+01;
    COFD[1769] = 5.53950393E+00;
    COFD[1770] = -4.86376204E-01;
    COFD[1771] = 2.03760106E-02;
    COFD[1772] = -2.09490548E+01;
    COFD[1773] = 5.31360223E+00;
    COFD[1774] = -4.64787000E-01;
    COFD[1775] = 1.97483720E-02;
    COFD[1776] = -2.09108261E+01;
    COFD[1777] = 5.30526648E+00;
    COFD[1778] = -4.63785596E-01;
    COFD[1779] = 1.97079873E-02;
    COFD[1780] = -2.09127554E+01;
    COFD[1781] = 5.30526648E+00;
    COFD[1782] = -4.63785596E-01;
    COFD[1783] = 1.97079873E-02;
    COFD[1784] = -2.19550907E+01;
    COFD[1785] = 5.49350509E+00;
    COFD[1786] = -4.81613405E-01;
    COFD[1787] = 2.02171734E-02;
    COFD[1788] = -2.19557531E+01;
    COFD[1789] = 5.49350509E+00;
    COFD[1790] = -4.81613405E-01;
    COFD[1791] = 2.02171734E-02;
    COFD[1792] = -1.59327297E+01;
    COFD[1793] = 3.65620899E+00;
    COFD[1794] = -2.62933804E-01;
    COFD[1795] = 1.15253223E-02;
    COFD[1796] = -1.50270339E+01;
    COFD[1797] = 3.46140064E+00;
    COFD[1798] = -2.38440092E-01;
    COFD[1799] = 1.04960087E-02;
    COFD[1800] = -1.24693568E+01;
    COFD[1801] = 2.76686648E+00;
    COFD[1802] = -1.49120141E-01;
    COFD[1803] = 6.66220432E-03;
    COFD[1804] = -1.57199037E+01;
    COFD[1805] = 4.19936335E+00;
    COFD[1806] = -3.32311009E-01;
    COFD[1807] = 1.44921003E-02;
    COFD[1808] = -1.50420953E+01;
    COFD[1809] = 3.46140064E+00;
    COFD[1810] = -2.38440092E-01;
    COFD[1811] = 1.04960087E-02;
    COFD[1812] = -2.14087397E+01;
    COFD[1813] = 5.57282008E+00;
    COFD[1814] = -4.76690890E-01;
    COFD[1815] = 1.94000719E-02;
    COFD[1816] = -1.62724462E+01;
    COFD[1817] = 3.79163564E+00;
    COFD[1818] = -2.80257365E-01;
    COFD[1819] = 1.22656902E-02;
    COFD[1820] = -1.62775714E+01;
    COFD[1821] = 3.79163564E+00;
    COFD[1822] = -2.80257365E-01;
    COFD[1823] = 1.22656902E-02;
    COFD[1824] = -1.62824412E+01;
    COFD[1825] = 3.79163564E+00;
    COFD[1826] = -2.80257365E-01;
    COFD[1827] = 1.22656902E-02;
    COFD[1828] = -1.92867554E+01;
    COFD[1829] = 4.83375900E+00;
    COFD[1830] = -4.09146560E-01;
    COFD[1831] = 1.76006599E-02;
    COFD[1832] = -1.59525102E+01;
    COFD[1833] = 3.66023858E+00;
    COFD[1834] = -2.63401043E-01;
    COFD[1835] = 1.15432000E-02;
    COFD[1836] = -2.14082453E+01;
    COFD[1837] = 5.55346617E+00;
    COFD[1838] = -4.87783156E-01;
    COFD[1839] = 2.04210886E-02;
    COFD[1840] = -1.72556729E+01;
    COFD[1841] = 4.19029808E+00;
    COFD[1842] = -3.31177076E-01;
    COFD[1843] = 1.44446234E-02;
    COFD[1844] = -1.72738845E+01;
    COFD[1845] = 4.19029808E+00;
    COFD[1846] = -3.31177076E-01;
    COFD[1847] = 1.44446234E-02;
    COFD[1848] = -1.72167708E+01;
    COFD[1849] = 4.16886779E+00;
    COFD[1850] = -3.28518156E-01;
    COFD[1851] = 1.43341626E-02;
    COFD[1852] = -2.13319784E+01;
    COFD[1853] = 5.52422470E+00;
    COFD[1854] = -4.84872944E-01;
    COFD[1855] = 2.03298213E-02;
    COFD[1856] = -1.94186547E+01;
    COFD[1857] = 4.84669430E+00;
    COFD[1858] = -4.10571455E-01;
    COFD[1859] = 1.76520543E-02;
    COFD[1860] = -2.11309207E+01;
    COFD[1861] = 5.41773516E+00;
    COFD[1862] = -4.73414338E-01;
    COFD[1863] = 1.99258685E-02;
    COFD[1864] = -2.11430338E+01;
    COFD[1865] = 5.41773516E+00;
    COFD[1866] = -4.73414338E-01;
    COFD[1867] = 1.99258685E-02;
    COFD[1868] = -2.13881945E+01;
    COFD[1869] = 5.52422470E+00;
    COFD[1870] = -4.84872944E-01;
    COFD[1871] = 2.03298213E-02;
    COFD[1872] = -1.95154079E+01;
    COFD[1873] = 4.94787350E+00;
    COFD[1874] = -4.22829292E-01;
    COFD[1875] = 1.81487163E-02;
    COFD[1876] = -2.02318658E+01;
    COFD[1877] = 5.12963391E+00;
    COFD[1878] = -4.44146826E-01;
    COFD[1879] = 1.89829640E-02;
    COFD[1880] = -1.90692595E+01;
    COFD[1881] = 4.80830699E+00;
    COFD[1882] = -4.06171933E-01;
    COFD[1883] = 1.74848791E-02;
    COFD[1884] = -2.11341653E+01;
    COFD[1885] = 5.41773516E+00;
    COFD[1886] = -4.73414338E-01;
    COFD[1887] = 1.99258685E-02;
    COFD[1888] = -2.03775651E+01;
    COFD[1889] = 5.16159436E+00;
    COFD[1890] = -4.46935283E-01;
    COFD[1891] = 1.90480297E-02;
    COFD[1892] = -2.03123540E+01;
    COFD[1893] = 5.14854169E+00;
    COFD[1894] = -4.45984343E-01;
    COFD[1895] = 1.90374217E-02;
    COFD[1896] = -2.18731920E+01;
    COFD[1897] = 5.55171660E+00;
    COFD[1898] = -4.87609504E-01;
    COFD[1899] = 2.04156590E-02;
    COFD[1900] = -2.08822487E+01;
    COFD[1901] = 5.28557747E+00;
    COFD[1902] = -4.61402384E-01;
    COFD[1903] = 1.96111546E-02;
    COFD[1904] = -2.08427678E+01;
    COFD[1905] = 5.27674330E+00;
    COFD[1906] = -4.60336155E-01;
    COFD[1907] = 1.95680191E-02;
    COFD[1908] = -2.08447974E+01;
    COFD[1909] = 5.27674330E+00;
    COFD[1910] = -4.60336155E-01;
    COFD[1911] = 1.95680191E-02;
    COFD[1912] = -2.19053841E+01;
    COFD[1913] = 5.47162499E+00;
    COFD[1914] = -4.79195552E-01;
    COFD[1915] = 2.01289088E-02;
    COFD[1916] = -2.19060847E+01;
    COFD[1917] = 5.47162499E+00;
    COFD[1918] = -4.79195552E-01;
    COFD[1919] = 2.01289088E-02;
    COFD[1920] = -2.03844252E+01;
    COFD[1921] = 5.18856872E+00;
    COFD[1922] = -4.50001829E-01;
    COFD[1923] = 1.91636142E-02;
    COFD[1924] = -1.93364585E+01;
    COFD[1925] = 4.98286777E+00;
    COFD[1926] = -4.26970814E-01;
    COFD[1927] = 1.83122917E-02;
    COFD[1928] = -1.59537247E+01;
    COFD[1929] = 4.07051484E+00;
    COFD[1930] = -3.16303109E-01;
    COFD[1931] = 1.38259377E-02;
    COFD[1932] = -1.96866103E+01;
    COFD[1933] = 5.54637286E+00;
    COFD[1934] = -4.87070324E-01;
    COFD[1935] = 2.03983467E-02;
    COFD[1936] = -1.93566243E+01;
    COFD[1937] = 4.98286777E+00;
    COFD[1938] = -4.26970814E-01;
    COFD[1939] = 1.83122917E-02;
    COFD[1940] = -1.80253664E+01;
    COFD[1941] = 3.69199168E+00;
    COFD[1942] = -1.74005516E-01;
    COFD[1943] = 3.97694372E-03;
    COFD[1944] = -2.07595845E+01;
    COFD[1945] = 5.32244593E+00;
    COFD[1946] = -4.65829403E-01;
    COFD[1947] = 1.97895274E-02;
    COFD[1948] = -2.07672833E+01;
    COFD[1949] = 5.32244593E+00;
    COFD[1950] = -4.65829403E-01;
    COFD[1951] = 1.97895274E-02;
    COFD[1952] = -2.07746356E+01;
    COFD[1953] = 5.32244593E+00;
    COFD[1954] = -4.65829403E-01;
    COFD[1955] = 1.97895274E-02;
    COFD[1956] = -2.20063594E+01;
    COFD[1957] = 5.48540187E+00;
    COFD[1958] = -4.58962148E-01;
    COFD[1959] = 1.83770355E-02;
    COFD[1960] = -2.04144604E+01;
    COFD[1961] = 5.19614628E+00;
    COFD[1962] = -4.50889164E-01;
    COFD[1963] = 1.91983328E-02;
    COFD[1964] = -1.93214527E+01;
    COFD[1965] = 4.10954793E+00;
    COFD[1966] = -2.37523329E-01;
    COFD[1967] = 7.08858141E-03;
    COFD[1968] = -2.13538553E+01;
    COFD[1969] = 5.54007827E+00;
    COFD[1970] = -4.86434511E-01;
    COFD[1971] = 2.03779006E-02;
    COFD[1972] = -2.13777308E+01;
    COFD[1973] = 5.54007827E+00;
    COFD[1974] = -4.86434511E-01;
    COFD[1975] = 2.03779006E-02;
    COFD[1976] = -2.13319784E+01;
    COFD[1977] = 5.52422470E+00;
    COFD[1978] = -4.84872944E-01;
    COFD[1979] = 2.03298213E-02;
    COFD[1980] = -1.95785144E+01;
    COFD[1981] = 4.22062499E+00;
    COFD[1982] = -2.54326872E-01;
    COFD[1983] = 7.91017784E-03;
    COFD[1984] = -2.20495822E+01;
    COFD[1985] = 5.47072190E+00;
    COFD[1986] = -4.56301261E-01;
    COFD[1987] = 1.82313566E-02;
    COFD[1988] = -2.03036402E+01;
    COFD[1989] = 4.50250781E+00;
    COFD[1990] = -2.97622106E-01;
    COFD[1991] = 1.00481473E-02;
    COFD[1992] = -2.03227406E+01;
    COFD[1993] = 4.50250781E+00;
    COFD[1994] = -2.97622106E-01;
    COFD[1995] = 1.00481473E-02;
    COFD[1996] = -1.96653154E+01;
    COFD[1997] = 4.22062499E+00;
    COFD[1998] = -2.54326872E-01;
    COFD[1999] = 7.91017784E-03;
    COFD[2000] = -2.17547312E+01;
    COFD[2001] = 5.40298848E+00;
    COFD[2002] = -4.43954594E-01;
    COFD[2003] = 1.75542998E-02;
    COFD[2004] = -2.16936515E+01;
    COFD[2005] = 5.21869603E+00;
    COFD[2006] = -4.12084772E-01;
    COFD[2007] = 1.58573035E-02;
    COFD[2008] = -2.18356866E+01;
    COFD[2009] = 5.49906960E+00;
    COFD[2010] = -4.61793001E-01;
    COFD[2011] = 1.85415189E-02;
    COFD[2012] = -2.03087302E+01;
    COFD[2013] = 4.50250781E+00;
    COFD[2014] = -2.97622106E-01;
    COFD[2015] = 1.00481473E-02;
    COFD[2016] = -2.15816909E+01;
    COFD[2017] = 5.13708607E+00;
    COFD[2018] = -3.98445708E-01;
    COFD[2019] = 1.51455626E-02;
    COFD[2020] = -2.16420936E+01;
    COFD[2021] = 5.17945041E+00;
    COFD[2022] = -4.05514689E-01;
    COFD[2023] = 1.55141412E-02;
    COFD[2024] = -2.00981944E+01;
    COFD[2025] = 4.22278378E+00;
    COFD[2026] = -2.54653500E-01;
    COFD[2027] = 7.92616085E-03;
    COFD[2028] = -2.13985484E+01;
    COFD[2029] = 4.94878244E+00;
    COFD[2030] = -3.68158605E-01;
    COFD[2031] = 1.36008797E-02;
    COFD[2032] = -2.14111310E+01;
    COFD[2033] = 4.96219227E+00;
    COFD[2034] = -3.70270843E-01;
    COFD[2035] = 1.37072211E-02;
    COFD[2036] = -2.14144448E+01;
    COFD[2037] = 4.96219227E+00;
    COFD[2038] = -3.70270843E-01;
    COFD[2039] = 1.37072211E-02;
    COFD[2040] = -2.06858147E+01;
    COFD[2041] = 4.35920123E+00;
    COFD[2042] = -2.75491273E-01;
    COFD[2043] = 8.95100289E-03;
    COFD[2044] = -2.06870442E+01;
    COFD[2045] = 4.35920123E+00;
    COFD[2046] = -2.75491273E-01;
    COFD[2047] = 8.95100289E-03;
    COFD[2048] = -1.82673770E+01;
    COFD[2049] = 4.39538102E+00;
    COFD[2050] = -3.56367230E-01;
    COFD[2051] = 1.54788461E-02;
    COFD[2052] = -1.72112971E+01;
    COFD[2053] = 4.15807461E+00;
    COFD[2054] = -3.27178539E-01;
    COFD[2055] = 1.42784349E-02;
    COFD[2056] = -1.39658996E+01;
    COFD[2057] = 3.24966086E+00;
    COFD[2058] = -2.11199992E-01;
    COFD[2059] = 9.32580661E-03;
    COFD[2060] = -1.78637178E+01;
    COFD[2061] = 4.88268692E+00;
    COFD[2062] = -4.14917638E-01;
    COFD[2063] = 1.78274298E-02;
    COFD[2064] = -1.72310232E+01;
    COFD[2065] = 4.15807461E+00;
    COFD[2066] = -3.27178539E-01;
    COFD[2067] = 1.42784349E-02;
    COFD[2068] = -2.13148887E+01;
    COFD[2069] = 5.27210469E+00;
    COFD[2070] = -4.21419216E-01;
    COFD[2071] = 1.63567178E-02;
    COFD[2072] = -1.85844688E+01;
    COFD[2073] = 4.51052425E+00;
    COFD[2074] = -3.70301627E-01;
    COFD[2075] = 1.60416153E-02;
    COFD[2076] = -1.85919214E+01;
    COFD[2077] = 4.51052425E+00;
    COFD[2078] = -3.70301627E-01;
    COFD[2079] = 1.60416153E-02;
    COFD[2080] = -1.85990352E+01;
    COFD[2081] = 4.51052425E+00;
    COFD[2082] = -3.70301627E-01;
    COFD[2083] = 1.60416153E-02;
    COFD[2084] = -2.14151520E+01;
    COFD[2085] = 5.41122754E+00;
    COFD[2086] = -4.73185889E-01;
    COFD[2087] = 1.99407905E-02;
    COFD[2088] = -1.82955252E+01;
    COFD[2089] = 4.40289649E+00;
    COFD[2090] = -3.57289765E-01;
    COFD[2091] = 1.55166804E-02;
    COFD[2092] = -2.19786173E+01;
    COFD[2093] = 5.43750833E+00;
    COFD[2094] = -4.50273329E-01;
    COFD[2095] = 1.79013718E-02;
    COFD[2096] = -1.94585111E+01;
    COFD[2097] = 4.87180830E+00;
    COFD[2098] = -4.13582958E-01;
    COFD[2099] = 1.77726094E-02;
    COFD[2100] = -1.94819080E+01;
    COFD[2101] = 4.87180830E+00;
    COFD[2102] = -4.13582958E-01;
    COFD[2103] = 1.77726094E-02;
    COFD[2104] = -1.94186547E+01;
    COFD[2105] = 4.84669430E+00;
    COFD[2106] = -4.10571455E-01;
    COFD[2107] = 1.76520543E-02;
    COFD[2108] = -2.20495822E+01;
    COFD[2109] = 5.47072190E+00;
    COFD[2110] = -4.56301261E-01;
    COFD[2111] = 1.82313566E-02;
    COFD[2112] = -2.14907782E+01;
    COFD[2113] = 5.41585806E+00;
    COFD[2114] = -4.73359323E-01;
    COFD[2115] = 1.99310239E-02;
    COFD[2116] = -2.22429814E+01;
    COFD[2117] = 5.53139819E+00;
    COFD[2118] = -4.68828555E-01;
    COFD[2119] = 1.89597887E-02;
    COFD[2120] = -2.22613837E+01;
    COFD[2121] = 5.53139819E+00;
    COFD[2122] = -4.68828555E-01;
    COFD[2123] = 1.89597887E-02;
    COFD[2124] = -2.21333822E+01;
    COFD[2125] = 5.47072190E+00;
    COFD[2126] = -4.56301261E-01;
    COFD[2127] = 1.82313566E-02;
    COFD[2128] = -2.15206146E+01;
    COFD[2129] = 5.48426911E+00;
    COFD[2130] = -4.80606512E-01;
    COFD[2131] = 2.01811046E-02;
    COFD[2132] = -2.21343023E+01;
    COFD[2133] = 5.60010742E+00;
    COFD[2134] = -4.91597429E-01;
    COFD[2135] = 2.04987718E-02;
    COFD[2136] = -2.12014186E+01;
    COFD[2137] = 5.40060531E+00;
    COFD[2138] = -4.72449699E-01;
    COFD[2139] = 1.99345817E-02;
    COFD[2140] = -2.22478879E+01;
    COFD[2141] = 5.53139819E+00;
    COFD[2142] = -4.68828555E-01;
    COFD[2143] = 1.89597887E-02;
    COFD[2144] = -2.22317182E+01;
    COFD[2145] = 5.61211818E+00;
    COFD[2146] = -4.91432482E-01;
    COFD[2147] = 2.04238731E-02;
    COFD[2148] = -2.21793326E+01;
    COFD[2149] = 5.60403905E+00;
    COFD[2150] = -4.91221691E-01;
    COFD[2151] = 2.04473483E-02;
    COFD[2152] = -2.25302512E+01;
    COFD[2153] = 5.47136127E+00;
    COFD[2154] = -4.56417141E-01;
    COFD[2155] = 1.82376994E-02;
    COFD[2156] = -2.24120415E+01;
    COFD[2157] = 5.58744076E+00;
    COFD[2158] = -4.84489462E-01;
    COFD[2159] = 1.99733042E-02;
    COFD[2160] = -2.23993836E+01;
    COFD[2161] = 5.58952429E+00;
    COFD[2162] = -4.85012530E-01;
    COFD[2163] = 2.00062142E-02;
    COFD[2164] = -2.24025650E+01;
    COFD[2165] = 5.58952429E+00;
    COFD[2166] = -4.85012530E-01;
    COFD[2167] = 2.00062142E-02;
    COFD[2168] = -2.28446667E+01;
    COFD[2169] = 5.50134401E+00;
    COFD[2170] = -4.62488197E-01;
    COFD[2171] = 1.85873697E-02;
    COFD[2172] = -2.28458380E+01;
    COFD[2173] = 5.50134401E+00;
    COFD[2174] = -4.62488197E-01;
    COFD[2175] = 1.85873697E-02;
    COFD[2176] = -2.02646611E+01;
    COFD[2177] = 5.10426133E+00;
    COFD[2178] = -4.41256919E-01;
    COFD[2179] = 1.88737290E-02;
    COFD[2180] = -1.90883268E+01;
    COFD[2181] = 4.84384483E+00;
    COFD[2182] = -4.10265575E-01;
    COFD[2183] = 1.76414287E-02;
    COFD[2184] = -1.57034851E+01;
    COFD[2185] = 3.93614244E+00;
    COFD[2186] = -2.99111497E-01;
    COFD[2187] = 1.30888229E-02;
    COFD[2188] = -1.94688688E+01;
    COFD[2189] = 5.43830787E+00;
    COFD[2190] = -4.75472880E-01;
    COFD[2191] = 1.99909996E-02;
    COFD[2192] = -1.91102652E+01;
    COFD[2193] = 4.84384483E+00;
    COFD[2194] = -4.10265575E-01;
    COFD[2195] = 1.76414287E-02;
    COFD[2196] = -1.87383952E+01;
    COFD[2197] = 3.96926341E+00;
    COFD[2198] = -2.16412264E-01;
    COFD[2199] = 6.06012078E-03;
    COFD[2200] = -2.05184870E+01;
    COFD[2201] = 5.18417470E+00;
    COFD[2202] = -4.49491573E-01;
    COFD[2203] = 1.91438508E-02;
    COFD[2204] = -2.05272328E+01;
    COFD[2205] = 5.18417470E+00;
    COFD[2206] = -4.49491573E-01;
    COFD[2207] = 1.91438508E-02;
    COFD[2208] = -2.05356023E+01;
    COFD[2209] = 5.18417470E+00;
    COFD[2210] = -4.49491573E-01;
    COFD[2211] = 1.91438508E-02;
    COFD[2212] = -2.22116706E+01;
    COFD[2213] = 5.54251230E+00;
    COFD[2214] = -4.70946314E-01;
    COFD[2215] = 1.90785869E-02;
    COFD[2216] = -2.02922701E+01;
    COFD[2217] = 5.11106992E+00;
    COFD[2218] = -4.42047129E-01;
    COFD[2219] = 1.89042990E-02;
    COFD[2220] = -2.01015340E+01;
    COFD[2221] = 4.41511629E+00;
    COFD[2222] = -2.84086963E-01;
    COFD[2223] = 9.37586971E-03;
    COFD[2224] = -2.11349086E+01;
    COFD[2225] = 5.42846112E+00;
    COFD[2226] = -4.74321870E-01;
    COFD[2227] = 1.99459749E-02;
    COFD[2228] = -2.11606963E+01;
    COFD[2229] = 5.42846112E+00;
    COFD[2230] = -4.74321870E-01;
    COFD[2231] = 1.99459749E-02;
    COFD[2232] = -2.11309207E+01;
    COFD[2233] = 5.41773516E+00;
    COFD[2234] = -4.73414338E-01;
    COFD[2235] = 1.99258685E-02;
    COFD[2236] = -2.03036402E+01;
    COFD[2237] = 4.50250781E+00;
    COFD[2238] = -2.97622106E-01;
    COFD[2239] = 1.00481473E-02;
    COFD[2240] = -2.22429814E+01;
    COFD[2241] = 5.53139819E+00;
    COFD[2242] = -4.68828555E-01;
    COFD[2243] = 1.89597887E-02;
    COFD[2244] = -2.09002742E+01;
    COFD[2245] = 4.72895031E+00;
    COFD[2246] = -3.33332771E-01;
    COFD[2247] = 1.18431478E-02;
    COFD[2248] = -2.09224206E+01;
    COFD[2249] = 4.72895031E+00;
    COFD[2250] = -3.33332771E-01;
    COFD[2251] = 1.18431478E-02;
    COFD[2252] = -2.04033972E+01;
    COFD[2253] = 4.50250781E+00;
    COFD[2254] = -2.97622106E-01;
    COFD[2255] = 1.00481473E-02;
    COFD[2256] = -2.20262793E+01;
    COFD[2257] = 5.49663315E+00;
    COFD[2258] = -4.61182837E-01;
    COFD[2259] = 1.85035558E-02;
    COFD[2260] = -2.20597305E+01;
    COFD[2261] = 5.34774760E+00;
    COFD[2262] = -4.34239753E-01;
    COFD[2263] = 1.70320676E-02;
    COFD[2264] = -2.20398328E+01;
    COFD[2265] = 5.56049839E+00;
    COFD[2266] = -4.74367872E-01;
    COFD[2267] = 1.92702787E-02;
    COFD[2268] = -2.09061629E+01;
    COFD[2269] = 4.72895031E+00;
    COFD[2270] = -3.33332771E-01;
    COFD[2271] = 1.18431478E-02;
    COFD[2272] = -2.19592125E+01;
    COFD[2273] = 5.27258289E+00;
    COFD[2274] = -4.21502790E-01;
    COFD[2275] = 1.63611949E-02;
    COFD[2276] = -2.20192352E+01;
    COFD[2277] = 5.31412694E+00;
    COFD[2278] = -4.28473898E-01;
    COFD[2279] = 1.67264841E-02;
    COFD[2280] = -2.08353693E+01;
    COFD[2281] = 4.50409026E+00;
    COFD[2282] = -2.97868419E-01;
    COFD[2283] = 1.00604224E-02;
    COFD[2284] = -2.19253091E+01;
    COFD[2285] = 5.14570932E+00;
    COFD[2286] = -3.99877142E-01;
    COFD[2287] = 1.52199557E-02;
    COFD[2288] = -2.19282979E+01;
    COFD[2289] = 5.15446948E+00;
    COFD[2290] = -4.01332769E-01;
    COFD[2291] = 1.52956262E-02;
    COFD[2292] = -2.19322003E+01;
    COFD[2293] = 5.15446948E+00;
    COFD[2294] = -4.01332769E-01;
    COFD[2295] = 1.52956262E-02;
    COFD[2296] = -2.13524540E+01;
    COFD[2297] = 4.61201872E+00;
    COFD[2298] = -3.14803338E-01;
    COFD[2299] = 1.09082984E-02;
    COFD[2300] = -2.13539532E+01;
    COFD[2301] = 4.61201872E+00;
    COFD[2302] = -3.14803338E-01;
    COFD[2303] = 1.09082984E-02;
    COFD[2304] = -2.02822946E+01;
    COFD[2305] = 5.10426133E+00;
    COFD[2306] = -4.41256919E-01;
    COFD[2307] = 1.88737290E-02;
    COFD[2308] = -1.91004157E+01;
    COFD[2309] = 4.84384483E+00;
    COFD[2310] = -4.10265575E-01;
    COFD[2311] = 1.76414287E-02;
    COFD[2312] = -1.57054717E+01;
    COFD[2313] = 3.93614244E+00;
    COFD[2314] = -2.99111497E-01;
    COFD[2315] = 1.30888229E-02;
    COFD[2316] = -1.94698843E+01;
    COFD[2317] = 5.43830787E+00;
    COFD[2318] = -4.75472880E-01;
    COFD[2319] = 1.99909996E-02;
    COFD[2320] = -1.91229033E+01;
    COFD[2321] = 4.84384483E+00;
    COFD[2322] = -4.10265575E-01;
    COFD[2323] = 1.76414287E-02;
    COFD[2324] = -1.87515645E+01;
    COFD[2325] = 3.96926341E+00;
    COFD[2326] = -2.16412264E-01;
    COFD[2327] = 6.06012078E-03;
    COFD[2328] = -2.05375724E+01;
    COFD[2329] = 5.18417470E+00;
    COFD[2330] = -4.49491573E-01;
    COFD[2331] = 1.91438508E-02;
    COFD[2332] = -2.05466616E+01;
    COFD[2333] = 5.18417470E+00;
    COFD[2334] = -4.49491573E-01;
    COFD[2335] = 1.91438508E-02;
    COFD[2336] = -2.05553656E+01;
    COFD[2337] = 5.18417470E+00;
    COFD[2338] = -4.49491573E-01;
    COFD[2339] = 1.91438508E-02;
    COFD[2340] = -2.22343363E+01;
    COFD[2341] = 5.54251230E+00;
    COFD[2342] = -4.70946314E-01;
    COFD[2343] = 1.90785869E-02;
    COFD[2344] = -2.03099025E+01;
    COFD[2345] = 5.11106992E+00;
    COFD[2346] = -4.42047129E-01;
    COFD[2347] = 1.89042990E-02;
    COFD[2348] = -2.01199204E+01;
    COFD[2349] = 4.41511629E+00;
    COFD[2350] = -2.84086963E-01;
    COFD[2351] = 9.37586971E-03;
    COFD[2352] = -2.11458678E+01;
    COFD[2353] = 5.42846112E+00;
    COFD[2354] = -4.74321870E-01;
    COFD[2355] = 1.99459749E-02;
    COFD[2356] = -2.11722423E+01;
    COFD[2357] = 5.42846112E+00;
    COFD[2358] = -4.74321870E-01;
    COFD[2359] = 1.99459749E-02;
    COFD[2360] = -2.11430338E+01;
    COFD[2361] = 5.41773516E+00;
    COFD[2362] = -4.73414338E-01;
    COFD[2363] = 1.99258685E-02;
    COFD[2364] = -2.03227406E+01;
    COFD[2365] = 4.50250781E+00;
    COFD[2366] = -2.97622106E-01;
    COFD[2367] = 1.00481473E-02;
    COFD[2368] = -2.22613837E+01;
    COFD[2369] = 5.53139819E+00;
    COFD[2370] = -4.68828555E-01;
    COFD[2371] = 1.89597887E-02;
    COFD[2372] = -2.09224206E+01;
    COFD[2373] = 4.72895031E+00;
    COFD[2374] = -3.33332771E-01;
    COFD[2375] = 1.18431478E-02;
    COFD[2376] = -2.09455936E+01;
    COFD[2377] = 4.72895031E+00;
    COFD[2378] = -3.33332771E-01;
    COFD[2379] = 1.18431478E-02;
    COFD[2380] = -2.04268153E+01;
    COFD[2381] = 4.50250781E+00;
    COFD[2382] = -2.97622106E-01;
    COFD[2383] = 1.00481473E-02;
    COFD[2384] = -2.20431319E+01;
    COFD[2385] = 5.49663315E+00;
    COFD[2386] = -4.61182837E-01;
    COFD[2387] = 1.85035558E-02;
    COFD[2388] = -2.20818886E+01;
    COFD[2389] = 5.34774760E+00;
    COFD[2390] = -4.34239753E-01;
    COFD[2391] = 1.70320676E-02;
    COFD[2392] = -2.20574820E+01;
    COFD[2393] = 5.56049839E+00;
    COFD[2394] = -4.74367872E-01;
    COFD[2395] = 1.92702787E-02;
    COFD[2396] = -2.09285776E+01;
    COFD[2397] = 4.72895031E+00;
    COFD[2398] = -3.33332771E-01;
    COFD[2399] = 1.18431478E-02;
    COFD[2400] = -2.19808152E+01;
    COFD[2401] = 5.27258289E+00;
    COFD[2402] = -4.21502790E-01;
    COFD[2403] = 1.63611949E-02;
    COFD[2404] = -2.20411190E+01;
    COFD[2405] = 5.31412694E+00;
    COFD[2406] = -4.28473898E-01;
    COFD[2407] = 1.67264841E-02;
    COFD[2408] = -2.08639466E+01;
    COFD[2409] = 4.50409026E+00;
    COFD[2410] = -2.97868419E-01;
    COFD[2411] = 1.00604224E-02;
    COFD[2412] = -2.19503032E+01;
    COFD[2413] = 5.14570932E+00;
    COFD[2414] = -3.99877142E-01;
    COFD[2415] = 1.52199557E-02;
    COFD[2416] = -2.19534987E+01;
    COFD[2417] = 5.15446948E+00;
    COFD[2418] = -4.01332769E-01;
    COFD[2419] = 1.52956262E-02;
    COFD[2420] = -2.19576037E+01;
    COFD[2421] = 5.15446948E+00;
    COFD[2422] = -4.01332769E-01;
    COFD[2423] = 1.52956262E-02;
    COFD[2424] = -2.13838498E+01;
    COFD[2425] = 4.61201872E+00;
    COFD[2426] = -3.14803338E-01;
    COFD[2427] = 1.09082984E-02;
    COFD[2428] = -2.13854464E+01;
    COFD[2429] = 4.61201872E+00;
    COFD[2430] = -3.14803338E-01;
    COFD[2431] = 1.09082984E-02;
    COFD[2432] = -2.04649069E+01;
    COFD[2433] = 5.18856872E+00;
    COFD[2434] = -4.50001829E-01;
    COFD[2435] = 1.91636142E-02;
    COFD[2436] = -1.93925667E+01;
    COFD[2437] = 4.98286777E+00;
    COFD[2438] = -4.26970814E-01;
    COFD[2439] = 1.83122917E-02;
    COFD[2440] = -1.59632479E+01;
    COFD[2441] = 4.07051484E+00;
    COFD[2442] = -3.16303109E-01;
    COFD[2443] = 1.38259377E-02;
    COFD[2444] = -1.96914944E+01;
    COFD[2445] = 5.54637286E+00;
    COFD[2446] = -4.87070324E-01;
    COFD[2447] = 2.03983467E-02;
    COFD[2448] = -1.94151822E+01;
    COFD[2449] = 4.98286777E+00;
    COFD[2450] = -4.26970814E-01;
    COFD[2451] = 1.83122917E-02;
    COFD[2452] = -1.80862867E+01;
    COFD[2453] = 3.69199168E+00;
    COFD[2454] = -1.74005516E-01;
    COFD[2455] = 3.97694372E-03;
    COFD[2456] = -2.08463209E+01;
    COFD[2457] = 5.32244593E+00;
    COFD[2458] = -4.65829403E-01;
    COFD[2459] = 1.97895274E-02;
    COFD[2460] = -2.08554914E+01;
    COFD[2461] = 5.32244593E+00;
    COFD[2462] = -4.65829403E-01;
    COFD[2463] = 1.97895274E-02;
    COFD[2464] = -2.08642748E+01;
    COFD[2465] = 5.32244593E+00;
    COFD[2466] = -4.65829403E-01;
    COFD[2467] = 1.97895274E-02;
    COFD[2468] = -2.21083035E+01;
    COFD[2469] = 5.48540187E+00;
    COFD[2470] = -4.58962148E-01;
    COFD[2471] = 1.83770355E-02;
    COFD[2472] = -2.04949373E+01;
    COFD[2473] = 5.19614628E+00;
    COFD[2474] = -4.50889164E-01;
    COFD[2475] = 1.91983328E-02;
    COFD[2476] = -1.94051843E+01;
    COFD[2477] = 4.10954793E+00;
    COFD[2478] = -2.37523329E-01;
    COFD[2479] = 7.08858141E-03;
    COFD[2480] = -2.14048982E+01;
    COFD[2481] = 5.54007827E+00;
    COFD[2482] = -4.86434511E-01;
    COFD[2483] = 2.03779006E-02;
    COFD[2484] = -2.14314090E+01;
    COFD[2485] = 5.54007827E+00;
    COFD[2486] = -4.86434511E-01;
    COFD[2487] = 2.03779006E-02;
    COFD[2488] = -2.13881945E+01;
    COFD[2489] = 5.52422470E+00;
    COFD[2490] = -4.84872944E-01;
    COFD[2491] = 2.03298213E-02;
    COFD[2492] = -1.96653154E+01;
    COFD[2493] = 4.22062499E+00;
    COFD[2494] = -2.54326872E-01;
    COFD[2495] = 7.91017784E-03;
    COFD[2496] = -2.21333822E+01;
    COFD[2497] = 5.47072190E+00;
    COFD[2498] = -4.56301261E-01;
    COFD[2499] = 1.82313566E-02;
    COFD[2500] = -2.04033972E+01;
    COFD[2501] = 4.50250781E+00;
    COFD[2502] = -2.97622106E-01;
    COFD[2503] = 1.00481473E-02;
    COFD[2504] = -2.04268153E+01;
    COFD[2505] = 4.50250781E+00;
    COFD[2506] = -2.97622106E-01;
    COFD[2507] = 1.00481473E-02;
    COFD[2508] = -1.97704178E+01;
    COFD[2509] = 4.22062499E+00;
    COFD[2510] = -2.54326872E-01;
    COFD[2511] = 7.91017784E-03;
    COFD[2512] = -2.18318278E+01;
    COFD[2513] = 5.40298848E+00;
    COFD[2514] = -4.43954594E-01;
    COFD[2515] = 1.75542998E-02;
    COFD[2516] = -2.17934580E+01;
    COFD[2517] = 5.21869603E+00;
    COFD[2518] = -4.12084772E-01;
    COFD[2519] = 1.58573035E-02;
    COFD[2520] = -2.19162360E+01;
    COFD[2521] = 5.49906960E+00;
    COFD[2522] = -4.61793001E-01;
    COFD[2523] = 1.85415189E-02;
    COFD[2524] = -2.04096182E+01;
    COFD[2525] = 4.50250781E+00;
    COFD[2526] = -2.97622106E-01;
    COFD[2527] = 1.00481473E-02;
    COFD[2528] = -2.16791513E+01;
    COFD[2529] = 5.13708607E+00;
    COFD[2530] = -3.98445708E-01;
    COFD[2531] = 1.51455626E-02;
    COFD[2532] = -2.17407419E+01;
    COFD[2533] = 5.17945041E+00;
    COFD[2534] = -4.05514689E-01;
    COFD[2535] = 1.55141412E-02;
    COFD[2536] = -2.02246117E+01;
    COFD[2537] = 4.22278378E+00;
    COFD[2538] = -2.54653500E-01;
    COFD[2539] = 7.92616085E-03;
    COFD[2540] = -2.15102238E+01;
    COFD[2541] = 4.94878244E+00;
    COFD[2542] = -3.68158605E-01;
    COFD[2543] = 1.36008797E-02;
    COFD[2544] = -2.15236645E+01;
    COFD[2545] = 4.96219227E+00;
    COFD[2546] = -3.70270843E-01;
    COFD[2547] = 1.37072211E-02;
    COFD[2548] = -2.15278182E+01;
    COFD[2549] = 4.96219227E+00;
    COFD[2550] = -3.70270843E-01;
    COFD[2551] = 1.37072211E-02;
    COFD[2552] = -2.08236367E+01;
    COFD[2553] = 4.35920123E+00;
    COFD[2554] = -2.75491273E-01;
    COFD[2555] = 8.95100289E-03;
    COFD[2556] = -2.08252570E+01;
    COFD[2557] = 4.35920123E+00;
    COFD[2558] = -2.75491273E-01;
    COFD[2559] = 8.95100289E-03;
    COFD[2560] = -1.83039618E+01;
    COFD[2561] = 4.47952077E+00;
    COFD[2562] = -3.66569471E-01;
    COFD[2563] = 1.58916129E-02;
    COFD[2564] = -1.72286007E+01;
    COFD[2565] = 4.24084025E+00;
    COFD[2566] = -3.37428619E-01;
    COFD[2567] = 1.47032793E-02;
    COFD[2568] = -1.39315266E+01;
    COFD[2569] = 3.30394764E+00;
    COFD[2570] = -2.17920112E-01;
    COFD[2571] = 9.60284243E-03;
    COFD[2572] = -1.79310765E+01;
    COFD[2573] = 4.98037650E+00;
    COFD[2574] = -4.26676911E-01;
    COFD[2575] = 1.83007231E-02;
    COFD[2576] = -1.72473011E+01;
    COFD[2577] = 4.24084025E+00;
    COFD[2578] = -3.37428619E-01;
    COFD[2579] = 1.47032793E-02;
    COFD[2580] = -2.09565916E+01;
    COFD[2581] = 5.18380539E+00;
    COFD[2582] = -4.06234719E-01;
    COFD[2583] = 1.55515345E-02;
    COFD[2584] = -1.86507213E+01;
    COFD[2585] = 4.60874797E+00;
    COFD[2586] = -3.82368716E-01;
    COFD[2587] = 1.65370164E-02;
    COFD[2588] = -1.86576191E+01;
    COFD[2589] = 4.60874797E+00;
    COFD[2590] = -3.82368716E-01;
    COFD[2591] = 1.65370164E-02;
    COFD[2592] = -1.86641962E+01;
    COFD[2593] = 4.60874797E+00;
    COFD[2594] = -3.82368716E-01;
    COFD[2595] = 1.65370164E-02;
    COFD[2596] = -2.13961414E+01;
    COFD[2597] = 5.46685775E+00;
    COFD[2598] = -4.78665416E-01;
    COFD[2599] = 2.01093915E-02;
    COFD[2600] = -1.83296965E+01;
    COFD[2601] = 4.48570999E+00;
    COFD[2602] = -3.67301524E-01;
    COFD[2603] = 1.59204254E-02;
    COFD[2604] = -2.16798265E+01;
    COFD[2605] = 5.36811769E+00;
    COFD[2606] = -4.37727086E-01;
    COFD[2607] = 1.72167686E-02;
    COFD[2608] = -1.95548230E+01;
    COFD[2609] = 4.97133070E+00;
    COFD[2610] = -4.25604177E-01;
    COFD[2611] = 1.82582594E-02;
    COFD[2612] = -1.95770968E+01;
    COFD[2613] = 4.97133070E+00;
    COFD[2614] = -4.25604177E-01;
    COFD[2615] = 1.82582594E-02;
    COFD[2616] = -1.95154079E+01;
    COFD[2617] = 4.94787350E+00;
    COFD[2618] = -4.22829292E-01;
    COFD[2619] = 1.81487163E-02;
    COFD[2620] = -2.17547312E+01;
    COFD[2621] = 5.40298848E+00;
    COFD[2622] = -4.43954594E-01;
    COFD[2623] = 1.75542998E-02;
    COFD[2624] = -2.15206146E+01;
    COFD[2625] = 5.48426911E+00;
    COFD[2626] = -4.80606512E-01;
    COFD[2627] = 2.01811046E-02;
    COFD[2628] = -2.20262793E+01;
    COFD[2629] = 5.49663315E+00;
    COFD[2630] = -4.61182837E-01;
    COFD[2631] = 1.85035558E-02;
    COFD[2632] = -2.20431319E+01;
    COFD[2633] = 5.49663315E+00;
    COFD[2634] = -4.61182837E-01;
    COFD[2635] = 1.85035558E-02;
    COFD[2636] = -2.18318278E+01;
    COFD[2637] = 5.40298848E+00;
    COFD[2638] = -4.43954594E-01;
    COFD[2639] = 1.75542998E-02;
    COFD[2640] = -2.15453676E+01;
    COFD[2641] = 5.55313619E+00;
    COFD[2642] = -4.87753729E-01;
    COFD[2643] = 2.04203421E-02;
    COFD[2644] = -2.20228343E+01;
    COFD[2645] = 5.61211028E+00;
    COFD[2646] = -4.90893171E-01;
    COFD[2647] = 2.03793118E-02;
    COFD[2648] = -2.11427744E+01;
    COFD[2649] = 5.43893233E+00;
    COFD[2650] = -4.75546039E-01;
    COFD[2651] = 1.99938690E-02;
    COFD[2652] = -2.20307777E+01;
    COFD[2653] = 5.49663315E+00;
    COFD[2654] = -4.61182837E-01;
    COFD[2655] = 1.85035558E-02;
    COFD[2656] = -2.20606550E+01;
    COFD[2657] = 5.59649805E+00;
    COFD[2658] = -4.86750336E-01;
    COFD[2659] = 2.01151498E-02;
    COFD[2660] = -2.20511271E+01;
    COFD[2661] = 5.60809037E+00;
    COFD[2662] = -4.89400803E-01;
    COFD[2663] = 2.02760802E-02;
    COFD[2664] = -2.22462130E+01;
    COFD[2665] = 5.40356304E+00;
    COFD[2666] = -4.44060256E-01;
    COFD[2667] = 1.75601121E-02;
    COFD[2668] = -2.22801170E+01;
    COFD[2669] = 5.58507108E+00;
    COFD[2670] = -4.81395065E-01;
    COFD[2671] = 1.97276199E-02;
    COFD[2672] = -2.22609256E+01;
    COFD[2673] = 5.58490856E+00;
    COFD[2674] = -4.81588720E-01;
    COFD[2675] = 1.97445317E-02;
    COFD[2676] = -2.22638165E+01;
    COFD[2677] = 5.58490856E+00;
    COFD[2678] = -4.81588720E-01;
    COFD[2679] = 1.97445317E-02;
    COFD[2680] = -2.26089431E+01;
    COFD[2681] = 5.44867280E+00;
    COFD[2682] = -4.52284883E-01;
    COFD[2683] = 1.80110706E-02;
    COFD[2684] = -2.26099899E+01;
    COFD[2685] = 5.44867280E+00;
    COFD[2686] = -4.52284883E-01;
    COFD[2687] = 1.80110706E-02;
    COFD[2688] = -1.90859283E+01;
    COFD[2689] = 4.68079396E+00;
    COFD[2690] = -3.91231550E-01;
    COFD[2691] = 1.69021170E-02;
    COFD[2692] = -1.79361160E+01;
    COFD[2693] = 4.42139452E+00;
    COFD[2694] = -3.59567329E-01;
    COFD[2695] = 1.56103969E-02;
    COFD[2696] = -1.45715797E+01;
    COFD[2697] = 3.49477850E+00;
    COFD[2698] = -2.42635772E-01;
    COFD[2699] = 1.06721490E-02;
    COFD[2700] = -1.85748546E+01;
    COFD[2701] = 5.14789919E+00;
    COFD[2702] = -4.45930850E-01;
    COFD[2703] = 1.90363341E-02;
    COFD[2704] = -1.79580609E+01;
    COFD[2705] = 4.42139452E+00;
    COFD[2706] = -3.59567329E-01;
    COFD[2707] = 1.56103969E-02;
    COFD[2708] = -2.06310304E+01;
    COFD[2709] = 4.89289496E+00;
    COFD[2710] = -3.59346263E-01;
    COFD[2711] = 1.31570901E-02;
    COFD[2712] = -1.93917298E+01;
    COFD[2713] = 4.78708023E+00;
    COFD[2714] = -4.03693144E-01;
    COFD[2715] = 1.73884817E-02;
    COFD[2716] = -1.94004795E+01;
    COFD[2717] = 4.78708023E+00;
    COFD[2718] = -4.03693144E-01;
    COFD[2719] = 1.73884817E-02;
    COFD[2720] = -1.94088529E+01;
    COFD[2721] = 4.78708023E+00;
    COFD[2722] = -4.03693144E-01;
    COFD[2723] = 1.73884817E-02;
    COFD[2724] = -2.20725883E+01;
    COFD[2725] = 5.59642965E+00;
    COFD[2726] = -4.91577716E-01;
    COFD[2727] = 2.05159582E-02;
    COFD[2728] = -1.91118445E+01;
    COFD[2729] = 4.68715685E+00;
    COFD[2730] = -3.91979493E-01;
    COFD[2731] = 1.69314004E-02;
    COFD[2732] = -2.15802788E+01;
    COFD[2733] = 5.16868516E+00;
    COFD[2734] = -4.03721581E-01;
    COFD[2735] = 1.54206640E-02;
    COFD[2736] = -2.02434438E+01;
    COFD[2737] = 5.14418672E+00;
    COFD[2738] = -4.45631004E-01;
    COFD[2739] = 1.90308403E-02;
    COFD[2740] = -2.02692384E+01;
    COFD[2741] = 5.14418672E+00;
    COFD[2742] = -4.45631004E-01;
    COFD[2743] = 1.90308403E-02;
    COFD[2744] = -2.02318658E+01;
    COFD[2745] = 5.12963391E+00;
    COFD[2746] = -4.44146826E-01;
    COFD[2747] = 1.89829640E-02;
    COFD[2748] = -2.16936515E+01;
    COFD[2749] = 5.21869603E+00;
    COFD[2750] = -4.12084772E-01;
    COFD[2751] = 1.58573035E-02;
    COFD[2752] = -2.21343023E+01;
    COFD[2753] = 5.60010742E+00;
    COFD[2754] = -4.91597429E-01;
    COFD[2755] = 2.04987718E-02;
    COFD[2756] = -2.20597305E+01;
    COFD[2757] = 5.34774760E+00;
    COFD[2758] = -4.34239753E-01;
    COFD[2759] = 1.70320676E-02;
    COFD[2760] = -2.20818886E+01;
    COFD[2761] = 5.34774760E+00;
    COFD[2762] = -4.34239753E-01;
    COFD[2763] = 1.70320676E-02;
    COFD[2764] = -2.17934580E+01;
    COFD[2765] = 5.21869603E+00;
    COFD[2766] = -4.12084772E-01;
    COFD[2767] = 1.58573035E-02;
    COFD[2768] = -2.20228343E+01;
    COFD[2769] = 5.61211028E+00;
    COFD[2770] = -4.90893171E-01;
    COFD[2771] = 2.03793118E-02;
    COFD[2772] = -2.23318349E+01;
    COFD[2773] = 5.58508387E+00;
    COFD[2774] = -4.81385216E-01;
    COFD[2775] = 1.97267369E-02;
    COFD[2776] = -2.18222696E+01;
    COFD[2777] = 5.57940140E+00;
    COFD[2778] = -4.89964112E-01;
    COFD[2779] = 2.04689539E-02;
    COFD[2780] = -2.20656222E+01;
    COFD[2781] = 5.34774760E+00;
    COFD[2782] = -4.34239753E-01;
    COFD[2783] = 1.70320676E-02;
    COFD[2784] = -2.23996837E+01;
    COFD[2785] = 5.58325398E+00;
    COFD[2786] = -4.79084067E-01;
    COFD[2787] = 1.95452935E-02;
    COFD[2788] = -2.23689627E+01;
    COFD[2789] = 5.58513878E+00;
    COFD[2790] = -4.80389524E-01;
    COFD[2791] = 1.96438689E-02;
    COFD[2792] = -2.22169882E+01;
    COFD[2793] = 5.21950983E+00;
    COFD[2794] = -4.12223195E-01;
    COFD[2795] = 1.58645894E-02;
    COFD[2796] = -2.25041734E+01;
    COFD[2797] = 5.51797622E+00;
    COFD[2798] = -4.66229499E-01;
    COFD[2799] = 1.88128348E-02;
    COFD[2800] = -2.24965286E+01;
    COFD[2801] = 5.52198915E+00;
    COFD[2802] = -4.67014474E-01;
    COFD[2803] = 1.88574253E-02;
    COFD[2804] = -2.25004333E+01;
    COFD[2805] = 5.52198915E+00;
    COFD[2806] = -4.67014474E-01;
    COFD[2807] = 1.88574253E-02;
    COFD[2808] = -2.26029886E+01;
    COFD[2809] = 5.27383847E+00;
    COFD[2810] = -4.21722368E-01;
    COFD[2811] = 1.63729618E-02;
    COFD[2812] = -2.26044889E+01;
    COFD[2813] = 5.27383847E+00;
    COFD[2814] = -4.21722368E-01;
    COFD[2815] = 1.63729618E-02;
    COFD[2816] = -1.78815889E+01;
    COFD[2817] = 4.34347890E+00;
    COFD[2818] = -3.49890003E-01;
    COFD[2819] = 1.52083459E-02;
    COFD[2820] = -1.68343393E+01;
    COFD[2821] = 4.11954900E+00;
    COFD[2822] = -3.22470391E-01;
    COFD[2823] = 1.40859564E-02;
    COFD[2824] = -1.36336373E+01;
    COFD[2825] = 3.22088176E+00;
    COFD[2826] = -2.07623790E-01;
    COFD[2827] = 9.17771542E-03;
    COFD[2828] = -1.74407963E+01;
    COFD[2829] = 4.83580036E+00;
    COFD[2830] = -4.09383573E-01;
    COFD[2831] = 1.76098175E-02;
    COFD[2832] = -1.68535757E+01;
    COFD[2833] = 4.11954900E+00;
    COFD[2834] = -3.22470391E-01;
    COFD[2835] = 1.40859564E-02;
    COFD[2836] = -2.11309197E+01;
    COFD[2837] = 5.32644193E+00;
    COFD[2838] = -4.30581064E-01;
    COFD[2839] = 1.68379725E-02;
    COFD[2840] = -1.82145353E+01;
    COFD[2841] = 4.46848269E+00;
    COFD[2842] = -3.65269718E-01;
    COFD[2843] = 1.58407652E-02;
    COFD[2844] = -1.82217198E+01;
    COFD[2845] = 4.46848269E+00;
    COFD[2846] = -3.65269718E-01;
    COFD[2847] = 1.58407652E-02;
    COFD[2848] = -1.82285740E+01;
    COFD[2849] = 4.46848269E+00;
    COFD[2850] = -3.65269718E-01;
    COFD[2851] = 1.58407652E-02;
    COFD[2852] = -2.11031143E+01;
    COFD[2853] = 5.39439999E+00;
    COFD[2854] = -4.72050184E-01;
    COFD[2855] = 1.99336257E-02;
    COFD[2856] = -1.79116531E+01;
    COFD[2857] = 4.35148286E+00;
    COFD[2858] = -3.50886647E-01;
    COFD[2859] = 1.52498573E-02;
    COFD[2860] = -2.17855148E+01;
    COFD[2861] = 5.47519298E+00;
    COFD[2862] = -4.57113040E-01;
    COFD[2863] = 1.82758312E-02;
    COFD[2864] = -1.90996795E+01;
    COFD[2865] = 4.82869066E+00;
    COFD[2866] = -4.08564514E-01;
    COFD[2867] = 1.75784675E-02;
    COFD[2868] = -1.91225414E+01;
    COFD[2869] = 4.82869066E+00;
    COFD[2870] = -4.08564514E-01;
    COFD[2871] = 1.75784675E-02;
    COFD[2872] = -1.90692595E+01;
    COFD[2873] = 4.80830699E+00;
    COFD[2874] = -4.06171933E-01;
    COFD[2875] = 1.74848791E-02;
    COFD[2876] = -2.18356866E+01;
    COFD[2877] = 5.49906960E+00;
    COFD[2878] = -4.61793001E-01;
    COFD[2879] = 1.85415189E-02;
    COFD[2880] = -2.12014186E+01;
    COFD[2881] = 5.40060531E+00;
    COFD[2882] = -4.72449699E-01;
    COFD[2883] = 1.99345817E-02;
    COFD[2884] = -2.20398328E+01;
    COFD[2885] = 5.56049839E+00;
    COFD[2886] = -4.74367872E-01;
    COFD[2887] = 1.92702787E-02;
    COFD[2888] = -2.20574820E+01;
    COFD[2889] = 5.56049839E+00;
    COFD[2890] = -4.74367872E-01;
    COFD[2891] = 1.92702787E-02;
    COFD[2892] = -2.19162360E+01;
    COFD[2893] = 5.49906960E+00;
    COFD[2894] = -4.61793001E-01;
    COFD[2895] = 1.85415189E-02;
    COFD[2896] = -2.11427744E+01;
    COFD[2897] = 5.43893233E+00;
    COFD[2898] = -4.75546039E-01;
    COFD[2899] = 1.99938690E-02;
    COFD[2900] = -2.18222696E+01;
    COFD[2901] = 5.57940140E+00;
    COFD[2902] = -4.89964112E-01;
    COFD[2903] = 2.04689539E-02;
    COFD[2904] = -2.08820897E+01;
    COFD[2905] = 5.38250415E+00;
    COFD[2906] = -4.71144140E-01;
    COFD[2907] = 1.99199779E-02;
    COFD[2908] = -2.20445411E+01;
    COFD[2909] = 5.56049839E+00;
    COFD[2910] = -4.74367872E-01;
    COFD[2911] = 1.92702787E-02;
    COFD[2912] = -2.19501296E+01;
    COFD[2913] = 5.60255148E+00;
    COFD[2914] = -4.91366572E-01;
    COFD[2915] = 2.04670553E-02;
    COFD[2916] = -2.19032561E+01;
    COFD[2917] = 5.59794138E+00;
    COFD[2918] = -4.91684532E-01;
    COFD[2919] = 2.05170953E-02;
    COFD[2920] = -2.23434237E+01;
    COFD[2921] = 5.49927389E+00;
    COFD[2922] = -4.61845436E-01;
    COFD[2923] = 1.85448066E-02;
    COFD[2924] = -2.21913393E+01;
    COFD[2925] = 5.60175327E+00;
    COFD[2926] = -4.87953216E-01;
    COFD[2927] = 2.01882171E-02;
    COFD[2928] = -2.21792065E+01;
    COFD[2929] = 5.60465338E+00;
    COFD[2930] = -4.88572478E-01;
    COFD[2931] = 2.02248525E-02;
    COFD[2932] = -2.21822461E+01;
    COFD[2933] = 5.60465338E+00;
    COFD[2934] = -4.88572478E-01;
    COFD[2935] = 2.02248525E-02;
    COFD[2936] = -2.26579938E+01;
    COFD[2937] = 5.52001624E+00;
    COFD[2938] = -4.66629503E-01;
    COFD[2939] = 1.88355817E-02;
    COFD[2940] = -2.26591038E+01;
    COFD[2941] = 5.52001624E+00;
    COFD[2942] = -4.66629503E-01;
    COFD[2943] = 1.88355817E-02;
    COFD[2944] = -2.02693653E+01;
    COFD[2945] = 5.10426133E+00;
    COFD[2946] = -4.41256919E-01;
    COFD[2947] = 1.88737290E-02;
    COFD[2948] = -1.90915649E+01;
    COFD[2949] = 4.84384483E+00;
    COFD[2950] = -4.10265575E-01;
    COFD[2951] = 1.76414287E-02;
    COFD[2952] = -1.57040212E+01;
    COFD[2953] = 3.93614244E+00;
    COFD[2954] = -2.99111497E-01;
    COFD[2955] = 1.30888229E-02;
    COFD[2956] = -1.94691430E+01;
    COFD[2957] = 5.43830787E+00;
    COFD[2958] = -4.75472880E-01;
    COFD[2959] = 1.99909996E-02;
    COFD[2960] = -1.91136491E+01;
    COFD[2961] = 4.84384483E+00;
    COFD[2962] = -4.10265575E-01;
    COFD[2963] = 1.76414287E-02;
    COFD[2964] = -1.87419199E+01;
    COFD[2965] = 3.96926341E+00;
    COFD[2966] = -2.16412264E-01;
    COFD[2967] = 6.06012078E-03;
    COFD[2968] = -2.05235731E+01;
    COFD[2969] = 5.18417470E+00;
    COFD[2970] = -4.49491573E-01;
    COFD[2971] = 1.91438508E-02;
    COFD[2972] = -2.05324091E+01;
    COFD[2973] = 5.18417470E+00;
    COFD[2974] = -4.49491573E-01;
    COFD[2975] = 1.91438508E-02;
    COFD[2976] = -2.05408665E+01;
    COFD[2977] = 5.18417470E+00;
    COFD[2978] = -4.49491573E-01;
    COFD[2979] = 1.91438508E-02;
    COFD[2980] = -2.22176950E+01;
    COFD[2981] = 5.54251230E+00;
    COFD[2982] = -4.70946314E-01;
    COFD[2983] = 1.90785869E-02;
    COFD[2984] = -2.02969740E+01;
    COFD[2985] = 5.11106992E+00;
    COFD[2986] = -4.42047129E-01;
    COFD[2987] = 1.89042990E-02;
    COFD[2988] = -2.01064363E+01;
    COFD[2989] = 4.41511629E+00;
    COFD[2990] = -2.84086963E-01;
    COFD[2991] = 9.37586971E-03;
    COFD[2992] = -2.11378465E+01;
    COFD[2993] = 5.42846112E+00;
    COFD[2994] = -4.74321870E-01;
    COFD[2995] = 1.99459749E-02;
    COFD[2996] = -2.11637902E+01;
    COFD[2997] = 5.42846112E+00;
    COFD[2998] = -4.74321870E-01;
    COFD[2999] = 1.99459749E-02;
    COFD[3000] = -2.11341653E+01;
    COFD[3001] = 5.41773516E+00;
    COFD[3002] = -4.73414338E-01;
    COFD[3003] = 1.99258685E-02;
    COFD[3004] = -2.03087302E+01;
    COFD[3005] = 4.50250781E+00;
    COFD[3006] = -2.97622106E-01;
    COFD[3007] = 1.00481473E-02;
    COFD[3008] = -2.22478879E+01;
    COFD[3009] = 5.53139819E+00;
    COFD[3010] = -4.68828555E-01;
    COFD[3011] = 1.89597887E-02;
    COFD[3012] = -2.09061629E+01;
    COFD[3013] = 4.72895031E+00;
    COFD[3014] = -3.33332771E-01;
    COFD[3015] = 1.18431478E-02;
    COFD[3016] = -2.09285776E+01;
    COFD[3017] = 4.72895031E+00;
    COFD[3018] = -3.33332771E-01;
    COFD[3019] = 1.18431478E-02;
    COFD[3020] = -2.04096182E+01;
    COFD[3021] = 4.50250781E+00;
    COFD[3022] = -2.97622106E-01;
    COFD[3023] = 1.00481473E-02;
    COFD[3024] = -2.20307777E+01;
    COFD[3025] = 5.49663315E+00;
    COFD[3026] = -4.61182837E-01;
    COFD[3027] = 1.85035558E-02;
    COFD[3028] = -2.20656222E+01;
    COFD[3029] = 5.34774760E+00;
    COFD[3030] = -4.34239753E-01;
    COFD[3031] = 1.70320676E-02;
    COFD[3032] = -2.20445411E+01;
    COFD[3033] = 5.56049839E+00;
    COFD[3034] = -4.74367872E-01;
    COFD[3035] = 1.92702787E-02;
    COFD[3036] = -2.09121217E+01;
    COFD[3037] = 4.72895031E+00;
    COFD[3038] = -3.33332771E-01;
    COFD[3039] = 1.18431478E-02;
    COFD[3040] = -2.19649589E+01;
    COFD[3041] = 5.27258289E+00;
    COFD[3042] = -4.21502790E-01;
    COFD[3043] = 1.63611949E-02;
    COFD[3044] = -2.20250551E+01;
    COFD[3045] = 5.31412694E+00;
    COFD[3046] = -4.28473898E-01;
    COFD[3047] = 1.67264841E-02;
    COFD[3048] = -2.08429322E+01;
    COFD[3049] = 4.50409026E+00;
    COFD[3050] = -2.97868419E-01;
    COFD[3051] = 1.00604224E-02;
    COFD[3052] = -2.19319411E+01;
    COFD[3053] = 5.14570932E+00;
    COFD[3054] = -3.99877142E-01;
    COFD[3055] = 1.52199557E-02;
    COFD[3056] = -2.19349837E+01;
    COFD[3057] = 5.15446948E+00;
    COFD[3058] = -4.01332769E-01;
    COFD[3059] = 1.52956262E-02;
    COFD[3060] = -2.19389389E+01;
    COFD[3061] = 5.15446948E+00;
    COFD[3062] = -4.01332769E-01;
    COFD[3063] = 1.52956262E-02;
    COFD[3064] = -2.13607457E+01;
    COFD[3065] = 4.61201872E+00;
    COFD[3066] = -3.14803338E-01;
    COFD[3067] = 1.09082984E-02;
    COFD[3068] = -2.13622700E+01;
    COFD[3069] = 4.61201872E+00;
    COFD[3070] = -3.14803338E-01;
    COFD[3071] = 1.09082984E-02;
    COFD[3072] = -1.92783884E+01;
    COFD[3073] = 4.73660584E+00;
    COFD[3074] = -3.97704978E-01;
    COFD[3075] = 1.71514887E-02;
    COFD[3076] = -1.81499793E+01;
    COFD[3077] = 4.48398491E+00;
    COFD[3078] = -3.67097129E-01;
    COFD[3079] = 1.59123634E-02;
    COFD[3080] = -1.47725694E+01;
    COFD[3081] = 3.55444478E+00;
    COFD[3082] = -2.50272707E-01;
    COFD[3083] = 1.09990787E-02;
    COFD[3084] = -1.87647862E+01;
    COFD[3085] = 5.19146813E+00;
    COFD[3086] = -4.50340408E-01;
    COFD[3087] = 1.91768178E-02;
    COFD[3088] = -1.81716176E+01;
    COFD[3089] = 4.48398491E+00;
    COFD[3090] = -3.67097129E-01;
    COFD[3091] = 1.59123634E-02;
    COFD[3092] = -2.04397451E+01;
    COFD[3093] = 4.77398686E+00;
    COFD[3094] = -3.40522956E-01;
    COFD[3095] = 1.22072846E-02;
    COFD[3096] = -1.95875976E+01;
    COFD[3097] = 4.84393038E+00;
    COFD[3098] = -4.10274737E-01;
    COFD[3099] = 1.76417458E-02;
    COFD[3100] = -1.95961596E+01;
    COFD[3101] = 4.84393038E+00;
    COFD[3102] = -4.10274737E-01;
    COFD[3103] = 1.76417458E-02;
    COFD[3104] = -1.96043503E+01;
    COFD[3105] = 4.84393038E+00;
    COFD[3106] = -4.10274737E-01;
    COFD[3107] = 1.76417458E-02;
    COFD[3108] = -2.21697404E+01;
    COFD[3109] = 5.60807471E+00;
    COFD[3110] = -4.91339309E-01;
    COFD[3111] = 2.04365761E-02;
    COFD[3112] = -1.93064215E+01;
    COFD[3113] = 4.74387793E+00;
    COFD[3114] = -3.98574972E-01;
    COFD[3115] = 1.71862289E-02;
    COFD[3116] = -2.14453157E+01;
    COFD[3117] = 5.07680397E+00;
    COFD[3118] = -3.88612087E-01;
    COFD[3119] = 1.46395101E-02;
    COFD[3120] = -2.04054899E+01;
    COFD[3121] = 5.18271974E+00;
    COFD[3122] = -4.49323627E-01;
    COFD[3123] = 1.91373940E-02;
    COFD[3124] = -2.04309557E+01;
    COFD[3125] = 5.18271974E+00;
    COFD[3126] = -4.49323627E-01;
    COFD[3127] = 1.91373940E-02;
    COFD[3128] = -2.03775651E+01;
    COFD[3129] = 5.16159436E+00;
    COFD[3130] = -4.46935283E-01;
    COFD[3131] = 1.90480297E-02;
    COFD[3132] = -2.15816909E+01;
    COFD[3133] = 5.13708607E+00;
    COFD[3134] = -3.98445708E-01;
    COFD[3135] = 1.51455626E-02;
    COFD[3136] = -2.22317182E+01;
    COFD[3137] = 5.61211818E+00;
    COFD[3138] = -4.91432482E-01;
    COFD[3139] = 2.04238731E-02;
    COFD[3140] = -2.19592125E+01;
    COFD[3141] = 5.27258289E+00;
    COFD[3142] = -4.21502790E-01;
    COFD[3143] = 1.63611949E-02;
    COFD[3144] = -2.19808152E+01;
    COFD[3145] = 5.27258289E+00;
    COFD[3146] = -4.21502790E-01;
    COFD[3147] = 1.63611949E-02;
    COFD[3148] = -2.16791513E+01;
    COFD[3149] = 5.13708607E+00;
    COFD[3150] = -3.98445708E-01;
    COFD[3151] = 1.51455626E-02;
    COFD[3152] = -2.20606550E+01;
    COFD[3153] = 5.59649805E+00;
    COFD[3154] = -4.86750336E-01;
    COFD[3155] = 2.01151498E-02;
    COFD[3156] = -2.23996837E+01;
    COFD[3157] = 5.58325398E+00;
    COFD[3158] = -4.79084067E-01;
    COFD[3159] = 1.95452935E-02;
    COFD[3160] = -2.19501296E+01;
    COFD[3161] = 5.60255148E+00;
    COFD[3162] = -4.91366572E-01;
    COFD[3163] = 2.04670553E-02;
    COFD[3164] = -2.19649589E+01;
    COFD[3165] = 5.27258289E+00;
    COFD[3166] = -4.21502790E-01;
    COFD[3167] = 1.63611949E-02;
    COFD[3168] = -2.23915398E+01;
    COFD[3169] = 5.54890339E+00;
    COFD[3170] = -4.72166228E-01;
    COFD[3171] = 1.91470071E-02;
    COFD[3172] = -2.24083163E+01;
    COFD[3173] = 5.57115285E+00;
    COFD[3174] = -4.76363416E-01;
    COFD[3175] = 1.93814080E-02;
    COFD[3176] = -2.20946432E+01;
    COFD[3177] = 5.13809011E+00;
    COFD[3178] = -3.98612308E-01;
    COFD[3179] = 1.51542189E-02;
    COFD[3180] = -2.25219004E+01;
    COFD[3181] = 5.49554403E+00;
    COFD[3182] = -4.60936491E-01;
    COFD[3183] = 1.84887572E-02;
    COFD[3184] = -2.25102565E+01;
    COFD[3185] = 5.49776513E+00;
    COFD[3186] = -4.61463030E-01;
    COFD[3187] = 1.85209236E-02;
    COFD[3188] = -2.25140525E+01;
    COFD[3189] = 5.49776513E+00;
    COFD[3190] = -4.61463030E-01;
    COFD[3191] = 1.85209236E-02;
    COFD[3192] = -2.25160816E+01;
    COFD[3193] = 5.21003123E+00;
    COFD[3194] = -4.10612564E-01;
    COFD[3195] = 1.57798598E-02;
    COFD[3196] = -2.25175307E+01;
    COFD[3197] = 5.21003123E+00;
    COFD[3198] = -4.10612564E-01;
    COFD[3199] = 1.57798598E-02;
    COFD[3200] = -1.91796663E+01;
    COFD[3201] = 4.70714822E+00;
    COFD[3202] = -3.94261134E-01;
    COFD[3203] = 1.70175169E-02;
    COFD[3204] = -1.80480958E+01;
    COFD[3205] = 4.45434023E+00;
    COFD[3206] = -3.63584633E-01;
    COFD[3207] = 1.57739270E-02;
    COFD[3208] = -1.46719197E+01;
    COFD[3209] = 3.52400594E+00;
    COFD[3210] = -2.46379985E-01;
    COFD[3211] = 1.08326032E-02;
    COFD[3212] = -1.86493112E+01;
    COFD[3213] = 5.16040659E+00;
    COFD[3214] = -4.46843492E-01;
    COFD[3215] = 1.90466181E-02;
    COFD[3216] = -1.80698901E+01;
    COFD[3217] = 4.45434023E+00;
    COFD[3218] = -3.63584633E-01;
    COFD[3219] = 1.57739270E-02;
    COFD[3220] = -2.05372411E+01;
    COFD[3221] = 4.83379373E+00;
    COFD[3222] = -3.50008083E-01;
    COFD[3223] = 1.26863426E-02;
    COFD[3224] = -1.94912151E+01;
    COFD[3225] = 4.81575071E+00;
    COFD[3226] = -4.07042139E-01;
    COFD[3227] = 1.75187504E-02;
    COFD[3228] = -1.94998722E+01;
    COFD[3229] = 4.81575071E+00;
    COFD[3230] = -4.07042139E-01;
    COFD[3231] = 1.75187504E-02;
    COFD[3232] = -1.95081555E+01;
    COFD[3233] = 4.81575071E+00;
    COFD[3234] = -4.07042139E-01;
    COFD[3235] = 1.75187504E-02;
    COFD[3236] = -2.21216828E+01;
    COFD[3237] = 5.60203389E+00;
    COFD[3238] = -4.91444416E-01;
    COFD[3239] = 2.04761886E-02;
    COFD[3240] = -1.92044492E+01;
    COFD[3241] = 4.71304783E+00;
    COFD[3242] = -3.94942083E-01;
    COFD[3243] = 1.70435959E-02;
    COFD[3244] = -2.15258568E+01;
    COFD[3245] = 5.12799307E+00;
    COFD[3246] = -3.96938732E-01;
    COFD[3247] = 1.50673195E-02;
    COFD[3248] = -2.03111230E+01;
    COFD[3249] = 5.15740122E+00;
    COFD[3250] = -4.46644818E-01;
    COFD[3251] = 1.90459001E-02;
    COFD[3252] = -2.03367561E+01;
    COFD[3253] = 5.15740122E+00;
    COFD[3254] = -4.46644818E-01;
    COFD[3255] = 1.90459001E-02;
    COFD[3256] = -2.03123540E+01;
    COFD[3257] = 5.14854169E+00;
    COFD[3258] = -4.45984343E-01;
    COFD[3259] = 1.90374217E-02;
    COFD[3260] = -2.16420936E+01;
    COFD[3261] = 5.17945041E+00;
    COFD[3262] = -4.05514689E-01;
    COFD[3263] = 1.55141412E-02;
    COFD[3264] = -2.21793326E+01;
    COFD[3265] = 5.60403905E+00;
    COFD[3266] = -4.91221691E-01;
    COFD[3267] = 2.04473483E-02;
    COFD[3268] = -2.20192352E+01;
    COFD[3269] = 5.31412694E+00;
    COFD[3270] = -4.28473898E-01;
    COFD[3271] = 1.67264841E-02;
    COFD[3272] = -2.20411190E+01;
    COFD[3273] = 5.31412694E+00;
    COFD[3274] = -4.28473898E-01;
    COFD[3275] = 1.67264841E-02;
    COFD[3276] = -2.17407419E+01;
    COFD[3277] = 5.17945041E+00;
    COFD[3278] = -4.05514689E-01;
    COFD[3279] = 1.55141412E-02;
    COFD[3280] = -2.20511271E+01;
    COFD[3281] = 5.60809037E+00;
    COFD[3282] = -4.89400803E-01;
    COFD[3283] = 2.02760802E-02;
    COFD[3284] = -2.23689627E+01;
    COFD[3285] = 5.58513878E+00;
    COFD[3286] = -4.80389524E-01;
    COFD[3287] = 1.96438689E-02;
    COFD[3288] = -2.19032561E+01;
    COFD[3289] = 5.59794138E+00;
    COFD[3290] = -4.91684532E-01;
    COFD[3291] = 2.05170953E-02;
    COFD[3292] = -2.20250551E+01;
    COFD[3293] = 5.31412694E+00;
    COFD[3294] = -4.28473898E-01;
    COFD[3295] = 1.67264841E-02;
    COFD[3296] = -2.24083163E+01;
    COFD[3297] = 5.57115285E+00;
    COFD[3298] = -4.76363416E-01;
    COFD[3299] = 1.93814080E-02;
    COFD[3300] = -2.24021886E+01;
    COFD[3301] = 5.58364149E+00;
    COFD[3302] = -4.79184111E-01;
    COFD[3303] = 1.95516164E-02;
    COFD[3304] = -2.21603646E+01;
    COFD[3305] = 5.18050127E+00;
    COFD[3306] = -4.05688517E-01;
    COFD[3307] = 1.55231713E-02;
    COFD[3308] = -2.25060112E+01;
    COFD[3309] = 5.50327119E+00;
    COFD[3310] = -4.63087223E-01;
    COFD[3311] = 1.86271401E-02;
    COFD[3312] = -2.24931486E+01;
    COFD[3313] = 5.50509817E+00;
    COFD[3314] = -4.63572794E-01;
    COFD[3315] = 1.86581046E-02;
    COFD[3316] = -2.24969995E+01;
    COFD[3317] = 5.50509817E+00;
    COFD[3318] = -4.63572794E-01;
    COFD[3319] = 1.86581046E-02;
    COFD[3320] = -2.25635595E+01;
    COFD[3321] = 5.24330646E+00;
    COFD[3322] = -4.16370120E-01;
    COFD[3323] = 1.60860486E-02;
    COFD[3324] = -2.25650343E+01;
    COFD[3325] = 5.24330646E+00;
    COFD[3326] = -4.16370120E-01;
    COFD[3327] = 1.60860486E-02;
    COFD[3328] = -2.09943481E+01;
    COFD[3329] = 5.22468467E+00;
    COFD[3330] = -4.54220128E-01;
    COFD[3331] = 1.93281042E-02;
    COFD[3332] = -1.98296243E+01;
    COFD[3333] = 4.98207523E+00;
    COFD[3334] = -4.26877291E-01;
    COFD[3335] = 1.83086094E-02;
    COFD[3336] = -1.64819183E+01;
    COFD[3337] = 4.11726215E+00;
    COFD[3338] = -3.22193015E-01;
    COFD[3339] = 1.40747074E-02;
    COFD[3340] = -2.01262921E+01;
    COFD[3341] = 5.54581286E+00;
    COFD[3342] = -4.87014004E-01;
    COFD[3343] = 2.03965482E-02;
    COFD[3344] = -1.98546695E+01;
    COFD[3345] = 4.98207523E+00;
    COFD[3346] = -4.26877291E-01;
    COFD[3347] = 1.83086094E-02;
    COFD[3348] = -1.73636900E+01;
    COFD[3349] = 3.17377130E+00;
    COFD[3350] = -1.00394383E-01;
    COFD[3351] = 5.69083899E-04;
    COFD[3352] = -2.13698722E+01;
    COFD[3353] = 5.34971865E+00;
    COFD[3354] = -4.68771123E-01;
    COFD[3355] = 1.98933811E-02;
    COFD[3356] = -2.12907159E+01;
    COFD[3357] = 5.32167660E+00;
    COFD[3358] = -4.65740624E-01;
    COFD[3359] = 1.97861081E-02;
    COFD[3360] = -2.13011157E+01;
    COFD[3361] = 5.32167660E+00;
    COFD[3362] = -4.65740624E-01;
    COFD[3363] = 1.97861081E-02;
    COFD[3364] = -2.25168081E+01;
    COFD[3365] = 5.46125558E+00;
    COFD[3366] = -4.54580949E-01;
    COFD[3367] = 1.81370928E-02;
    COFD[3368] = -2.10310742E+01;
    COFD[3369] = 5.23485505E+00;
    COFD[3370] = -4.55400362E-01;
    COFD[3371] = 1.93737680E-02;
    COFD[3372] = -1.98359760E+01;
    COFD[3373] = 4.11158627E+00;
    COFD[3374] = -2.37831519E-01;
    COFD[3375] = 7.10363413E-03;
    COFD[3376] = -2.17867314E+01;
    COFD[3377] = 5.53950393E+00;
    COFD[3378] = -4.86376204E-01;
    COFD[3379] = 2.03760106E-02;
    COFD[3380] = -2.18158049E+01;
    COFD[3381] = 5.53950393E+00;
    COFD[3382] = -4.86376204E-01;
    COFD[3383] = 2.03760106E-02;
    COFD[3384] = -2.18731920E+01;
    COFD[3385] = 5.55171660E+00;
    COFD[3386] = -4.87609504E-01;
    COFD[3387] = 2.04156590E-02;
    COFD[3388] = -2.00981944E+01;
    COFD[3389] = 4.22278378E+00;
    COFD[3390] = -2.54653500E-01;
    COFD[3391] = 7.92616085E-03;
    COFD[3392] = -2.25302512E+01;
    COFD[3393] = 5.47136127E+00;
    COFD[3394] = -4.56417141E-01;
    COFD[3395] = 1.82376994E-02;
    COFD[3396] = -2.08353693E+01;
    COFD[3397] = 4.50409026E+00;
    COFD[3398] = -2.97868419E-01;
    COFD[3399] = 1.00604224E-02;
    COFD[3400] = -2.08639466E+01;
    COFD[3401] = 4.50409026E+00;
    COFD[3402] = -2.97868419E-01;
    COFD[3403] = 1.00604224E-02;
    COFD[3404] = -2.02246117E+01;
    COFD[3405] = 4.22278378E+00;
    COFD[3406] = -2.54653500E-01;
    COFD[3407] = 7.92616085E-03;
    COFD[3408] = -2.22462130E+01;
    COFD[3409] = 5.40356304E+00;
    COFD[3410] = -4.44060256E-01;
    COFD[3411] = 1.75601121E-02;
    COFD[3412] = -2.22169882E+01;
    COFD[3413] = 5.21950983E+00;
    COFD[3414] = -4.12223195E-01;
    COFD[3415] = 1.58645894E-02;
    COFD[3416] = -2.23434237E+01;
    COFD[3417] = 5.49927389E+00;
    COFD[3418] = -4.61845436E-01;
    COFD[3419] = 1.85448066E-02;
    COFD[3420] = -2.08429322E+01;
    COFD[3421] = 4.50409026E+00;
    COFD[3422] = -2.97868419E-01;
    COFD[3423] = 1.00604224E-02;
    COFD[3424] = -2.20946432E+01;
    COFD[3425] = 5.13809011E+00;
    COFD[3426] = -3.98612308E-01;
    COFD[3427] = 1.51542189E-02;
    COFD[3428] = -2.21603646E+01;
    COFD[3429] = 5.18050127E+00;
    COFD[3430] = -4.05688517E-01;
    COFD[3431] = 1.55231713E-02;
    COFD[3432] = -2.02828056E+01;
    COFD[3433] = 4.06866060E+00;
    COFD[3434] = -2.33527101E-01;
    COFD[3435] = 6.97454219E-03;
    COFD[3436] = -2.19287691E+01;
    COFD[3437] = 4.95026695E+00;
    COFD[3438] = -3.68392434E-01;
    COFD[3439] = 1.36126514E-02;
    COFD[3440] = -2.19456782E+01;
    COFD[3441] = 4.96368178E+00;
    COFD[3442] = -3.70505465E-01;
    COFD[3443] = 1.37190339E-02;
    COFD[3444] = -2.19508859E+01;
    COFD[3445] = 4.96368178E+00;
    COFD[3446] = -3.70505465E-01;
    COFD[3447] = 1.37190339E-02;
    COFD[3448] = -2.12332312E+01;
    COFD[3449] = 4.36095377E+00;
    COFD[3450] = -2.75760539E-01;
    COFD[3451] = 8.96430249E-03;
    COFD[3452] = -2.12354028E+01;
    COFD[3453] = 4.36095377E+00;
    COFD[3454] = -2.75760539E-01;
    COFD[3455] = 8.96430249E-03;
    COFD[3456] = -1.97484166E+01;
    COFD[3457] = 4.84231878E+00;
    COFD[3458] = -4.10101001E-01;
    COFD[3459] = 1.76356687E-02;
    COFD[3460] = -1.86652603E+01;
    COFD[3461] = 4.61260432E+00;
    COFD[3462] = -3.82854484E-01;
    COFD[3463] = 1.65575163E-02;
    COFD[3464] = -1.51448279E+01;
    COFD[3465] = 3.64565939E+00;
    COFD[3466] = -2.61726871E-01;
    COFD[3467] = 1.14799244E-02;
    COFD[3468] = -1.92784178E+01;
    COFD[3469] = 5.32291505E+00;
    COFD[3470] = -4.65883522E-01;
    COFD[3471] = 1.97916109E-02;
    COFD[3472] = -1.86886689E+01;
    COFD[3473] = 4.61260432E+00;
    COFD[3474] = -3.82854484E-01;
    COFD[3475] = 1.65575163E-02;
    COFD[3476] = -2.02184916E+01;
    COFD[3477] = 4.57152878E+00;
    COFD[3478] = -3.08371263E-01;
    COFD[3479] = 1.05838559E-02;
    COFD[3480] = -2.01315602E+01;
    COFD[3481] = 4.97613338E+00;
    COFD[3482] = -4.26175206E-01;
    COFD[3483] = 1.82809270E-02;
    COFD[3484] = -2.01412473E+01;
    COFD[3485] = 4.97613338E+00;
    COFD[3486] = -4.26175206E-01;
    COFD[3487] = 1.82809270E-02;
    COFD[3488] = -2.01505348E+01;
    COFD[3489] = 4.97613338E+00;
    COFD[3490] = -4.26175206E-01;
    COFD[3491] = 1.82809270E-02;
    COFD[3492] = -2.23890317E+01;
    COFD[3493] = 5.59178974E+00;
    COFD[3494] = -4.85668031E-01;
    COFD[3495] = 2.00491907E-02;
    COFD[3496] = -1.97709603E+01;
    COFD[3497] = 4.84731557E+00;
    COFD[3498] = -4.10638352E-01;
    COFD[3499] = 1.76543886E-02;
    COFD[3500] = -2.12219677E+01;
    COFD[3501] = 4.87252053E+00;
    COFD[3502] = -3.56127804E-01;
    COFD[3503] = 1.29948788E-02;
    COFD[3504] = -2.09217020E+01;
    COFD[3505] = 5.31360223E+00;
    COFD[3506] = -4.64787000E-01;
    COFD[3507] = 1.97483720E-02;
    COFD[3508] = -2.09490548E+01;
    COFD[3509] = 5.31360223E+00;
    COFD[3510] = -4.64787000E-01;
    COFD[3511] = 1.97483720E-02;
    COFD[3512] = -2.08822487E+01;
    COFD[3513] = 5.28557747E+00;
    COFD[3514] = -4.61402384E-01;
    COFD[3515] = 1.96111546E-02;
    COFD[3516] = -2.13985484E+01;
    COFD[3517] = 4.94878244E+00;
    COFD[3518] = -3.68158605E-01;
    COFD[3519] = 1.36008797E-02;
    COFD[3520] = -2.24120415E+01;
    COFD[3521] = 5.58744076E+00;
    COFD[3522] = -4.84489462E-01;
    COFD[3523] = 1.99733042E-02;
    COFD[3524] = -2.19253091E+01;
    COFD[3525] = 5.14570932E+00;
    COFD[3526] = -3.99877142E-01;
    COFD[3527] = 1.52199557E-02;
    COFD[3528] = -2.19503032E+01;
    COFD[3529] = 5.14570932E+00;
    COFD[3530] = -3.99877142E-01;
    COFD[3531] = 1.52199557E-02;
    COFD[3532] = -2.15102238E+01;
    COFD[3533] = 4.94878244E+00;
    COFD[3534] = -3.68158605E-01;
    COFD[3535] = 1.36008797E-02;
    COFD[3536] = -2.22801170E+01;
    COFD[3537] = 5.58507108E+00;
    COFD[3538] = -4.81395065E-01;
    COFD[3539] = 1.97276199E-02;
    COFD[3540] = -2.25041734E+01;
    COFD[3541] = 5.51797622E+00;
    COFD[3542] = -4.66229499E-01;
    COFD[3543] = 1.88128348E-02;
    COFD[3544] = -2.21913393E+01;
    COFD[3545] = 5.60175327E+00;
    COFD[3546] = -4.87953216E-01;
    COFD[3547] = 2.01882171E-02;
    COFD[3548] = -2.19319411E+01;
    COFD[3549] = 5.14570932E+00;
    COFD[3550] = -3.99877142E-01;
    COFD[3551] = 1.52199557E-02;
    COFD[3552] = -2.25219004E+01;
    COFD[3553] = 5.49554403E+00;
    COFD[3554] = -4.60936491E-01;
    COFD[3555] = 1.84887572E-02;
    COFD[3556] = -2.25060112E+01;
    COFD[3557] = 5.50327119E+00;
    COFD[3558] = -4.63087223E-01;
    COFD[3559] = 1.86271401E-02;
    COFD[3560] = -2.19287691E+01;
    COFD[3561] = 4.95026695E+00;
    COFD[3562] = -3.68392434E-01;
    COFD[3563] = 1.36126514E-02;
    COFD[3564] = -2.25758616E+01;
    COFD[3565] = 5.40563818E+00;
    COFD[3566] = -4.44444322E-01;
    COFD[3567] = 1.75813146E-02;
    COFD[3568] = -2.25715533E+01;
    COFD[3569] = 5.41049872E+00;
    COFD[3570] = -4.45356411E-01;
    COFD[3571] = 1.76320470E-02;
    COFD[3572] = -2.25760230E+01;
    COFD[3573] = 5.41049872E+00;
    COFD[3574] = -4.45356411E-01;
    COFD[3575] = 1.76320470E-02;
    COFD[3576] = -2.24161979E+01;
    COFD[3577] = 5.05061421E+00;
    COFD[3578] = -3.84359196E-01;
    COFD[3579] = 1.44214004E-02;
    COFD[3580] = -2.24179759E+01;
    COFD[3581] = 5.05061421E+00;
    COFD[3582] = -3.84359196E-01;
    COFD[3583] = 1.44214004E-02;
    COFD[3584] = -1.97196489E+01;
    COFD[3585] = 4.83750266E+00;
    COFD[3586] = -4.09581452E-01;
    COFD[3587] = 1.76174739E-02;
    COFD[3588] = -1.86234701E+01;
    COFD[3589] = 4.60336076E+00;
    COFD[3590] = -3.81691643E-01;
    COFD[3591] = 1.65085234E-02;
    COFD[3592] = -1.51159870E+01;
    COFD[3593] = 3.64206330E+00;
    COFD[3594] = -2.61313444E-01;
    COFD[3595] = 1.14642754E-02;
    COFD[3596] = -1.92360228E+01;
    COFD[3597] = 5.31542554E+00;
    COFD[3598] = -4.65003780E-01;
    COFD[3599] = 1.97570185E-02;
    COFD[3600] = -1.86469792E+01;
    COFD[3601] = 4.60336076E+00;
    COFD[3602] = -3.81691643E-01;
    COFD[3603] = 1.65085234E-02;
    COFD[3604] = -2.02265558E+01;
    COFD[3605] = 4.58441724E+00;
    COFD[3606] = -3.10392854E-01;
    COFD[3607] = 1.06849990E-02;
    COFD[3608] = -2.00964665E+01;
    COFD[3609] = 4.96870443E+00;
    COFD[3610] = -4.25292447E-01;
    COFD[3611] = 1.82459096E-02;
    COFD[3612] = -2.01062206E+01;
    COFD[3613] = 4.96870443E+00;
    COFD[3614] = -4.25292447E-01;
    COFD[3615] = 1.82459096E-02;
    COFD[3616] = -2.01155735E+01;
    COFD[3617] = 4.96870443E+00;
    COFD[3618] = -4.25292447E-01;
    COFD[3619] = 1.82459096E-02;
    COFD[3620] = -2.23772680E+01;
    COFD[3621] = 5.59425354E+00;
    COFD[3622] = -4.86232980E-01;
    COFD[3623] = 2.00835981E-02;
    COFD[3624] = -1.97422209E+01;
    COFD[3625] = 4.84249900E+00;
    COFD[3626] = -4.10120448E-01;
    COFD[3627] = 1.76363500E-02;
    COFD[3628] = -2.12330900E+01;
    COFD[3629] = 4.88535789E+00;
    COFD[3630] = -3.58153894E-01;
    COFD[3631] = 1.30969624E-02;
    COFD[3632] = -2.08833669E+01;
    COFD[3633] = 5.30526648E+00;
    COFD[3634] = -4.63785596E-01;
    COFD[3635] = 1.97079873E-02;
    COFD[3636] = -2.09108261E+01;
    COFD[3637] = 5.30526648E+00;
    COFD[3638] = -4.63785596E-01;
    COFD[3639] = 1.97079873E-02;
    COFD[3640] = -2.08427678E+01;
    COFD[3641] = 5.27674330E+00;
    COFD[3642] = -4.60336155E-01;
    COFD[3643] = 1.95680191E-02;
    COFD[3644] = -2.14111310E+01;
    COFD[3645] = 4.96219227E+00;
    COFD[3646] = -3.70270843E-01;
    COFD[3647] = 1.37072211E-02;
    COFD[3648] = -2.23993836E+01;
    COFD[3649] = 5.58952429E+00;
    COFD[3650] = -4.85012530E-01;
    COFD[3651] = 2.00062142E-02;
    COFD[3652] = -2.19282979E+01;
    COFD[3653] = 5.15446948E+00;
    COFD[3654] = -4.01332769E-01;
    COFD[3655] = 1.52956262E-02;
    COFD[3656] = -2.19534987E+01;
    COFD[3657] = 5.15446948E+00;
    COFD[3658] = -4.01332769E-01;
    COFD[3659] = 1.52956262E-02;
    COFD[3660] = -2.15236645E+01;
    COFD[3661] = 4.96219227E+00;
    COFD[3662] = -3.70270843E-01;
    COFD[3663] = 1.37072211E-02;
    COFD[3664] = -2.22609256E+01;
    COFD[3665] = 5.58490856E+00;
    COFD[3666] = -4.81588720E-01;
    COFD[3667] = 1.97445317E-02;
    COFD[3668] = -2.24965286E+01;
    COFD[3669] = 5.52198915E+00;
    COFD[3670] = -4.67014474E-01;
    COFD[3671] = 1.88574253E-02;
    COFD[3672] = -2.21792065E+01;
    COFD[3673] = 5.60465338E+00;
    COFD[3674] = -4.88572478E-01;
    COFD[3675] = 2.02248525E-02;
    COFD[3676] = -2.19349837E+01;
    COFD[3677] = 5.15446948E+00;
    COFD[3678] = -4.01332769E-01;
    COFD[3679] = 1.52956262E-02;
    COFD[3680] = -2.25102565E+01;
    COFD[3681] = 5.49776513E+00;
    COFD[3682] = -4.61463030E-01;
    COFD[3683] = 1.85209236E-02;
    COFD[3684] = -2.24931486E+01;
    COFD[3685] = 5.50509817E+00;
    COFD[3686] = -4.63572794E-01;
    COFD[3687] = 1.86581046E-02;
    COFD[3688] = -2.19456782E+01;
    COFD[3689] = 4.96368178E+00;
    COFD[3690] = -3.70505465E-01;
    COFD[3691] = 1.37190339E-02;
    COFD[3692] = -2.25715533E+01;
    COFD[3693] = 5.41049872E+00;
    COFD[3694] = -4.45356411E-01;
    COFD[3695] = 1.76320470E-02;
    COFD[3696] = -2.25672005E+01;
    COFD[3697] = 5.41536807E+00;
    COFD[3698] = -4.46269562E-01;
    COFD[3699] = 1.76828228E-02;
    COFD[3700] = -2.25717119E+01;
    COFD[3701] = 5.41536807E+00;
    COFD[3702] = -4.46269562E-01;
    COFD[3703] = 1.76828228E-02;
    COFD[3704] = -2.24284563E+01;
    COFD[3705] = 5.06106414E+00;
    COFD[3706] = -3.86053039E-01;
    COFD[3707] = 1.45081784E-02;
    COFD[3708] = -2.24302556E+01;
    COFD[3709] = 5.06106414E+00;
    COFD[3710] = -3.86053039E-01;
    COFD[3711] = 1.45081784E-02;
    COFD[3712] = -1.97226856E+01;
    COFD[3713] = 4.83750266E+00;
    COFD[3714] = -4.09581452E-01;
    COFD[3715] = 1.76174739E-02;
    COFD[3716] = -1.86254955E+01;
    COFD[3717] = 4.60336076E+00;
    COFD[3718] = -3.81691643E-01;
    COFD[3719] = 1.65085234E-02;
    COFD[3720] = -1.51163041E+01;
    COFD[3721] = 3.64206330E+00;
    COFD[3722] = -2.61313444E-01;
    COFD[3723] = 1.14642754E-02;
    COFD[3724] = -1.92361841E+01;
    COFD[3725] = 5.31542554E+00;
    COFD[3726] = -4.65003780E-01;
    COFD[3727] = 1.97570185E-02;
    COFD[3728] = -1.86491023E+01;
    COFD[3729] = 4.60336076E+00;
    COFD[3730] = -3.81691643E-01;
    COFD[3731] = 1.65085234E-02;
    COFD[3732] = -2.02287739E+01;
    COFD[3733] = 4.58441724E+00;
    COFD[3734] = -3.10392854E-01;
    COFD[3735] = 1.06849990E-02;
    COFD[3736] = -2.00997774E+01;
    COFD[3737] = 4.96870443E+00;
    COFD[3738] = -4.25292447E-01;
    COFD[3739] = 1.82459096E-02;
    COFD[3740] = -2.01095969E+01;
    COFD[3741] = 4.96870443E+00;
    COFD[3742] = -4.25292447E-01;
    COFD[3743] = 1.82459096E-02;
    COFD[3744] = -2.01190139E+01;
    COFD[3745] = 4.96870443E+00;
    COFD[3746] = -4.25292447E-01;
    COFD[3747] = 1.82459096E-02;
    COFD[3748] = -2.23812726E+01;
    COFD[3749] = 5.59425354E+00;
    COFD[3750] = -4.86232980E-01;
    COFD[3751] = 2.00835981E-02;
    COFD[3752] = -1.97452574E+01;
    COFD[3753] = 4.84249900E+00;
    COFD[3754] = -4.10120448E-01;
    COFD[3755] = 1.76363500E-02;
    COFD[3756] = -2.12362684E+01;
    COFD[3757] = 4.88535789E+00;
    COFD[3758] = -3.58153894E-01;
    COFD[3759] = 1.30969624E-02;
    COFD[3760] = -2.08851929E+01;
    COFD[3761] = 5.30526648E+00;
    COFD[3762] = -4.63785596E-01;
    COFD[3763] = 1.97079873E-02;
    COFD[3764] = -2.09127554E+01;
    COFD[3765] = 5.30526648E+00;
    COFD[3766] = -4.63785596E-01;
    COFD[3767] = 1.97079873E-02;
    COFD[3768] = -2.08447974E+01;
    COFD[3769] = 5.27674330E+00;
    COFD[3770] = -4.60336155E-01;
    COFD[3771] = 1.95680191E-02;
    COFD[3772] = -2.14144448E+01;
    COFD[3773] = 4.96219227E+00;
    COFD[3774] = -3.70270843E-01;
    COFD[3775] = 1.37072211E-02;
    COFD[3776] = -2.24025650E+01;
    COFD[3777] = 5.58952429E+00;
    COFD[3778] = -4.85012530E-01;
    COFD[3779] = 2.00062142E-02;
    COFD[3780] = -2.19322003E+01;
    COFD[3781] = 5.15446948E+00;
    COFD[3782] = -4.01332769E-01;
    COFD[3783] = 1.52956262E-02;
    COFD[3784] = -2.19576037E+01;
    COFD[3785] = 5.15446948E+00;
    COFD[3786] = -4.01332769E-01;
    COFD[3787] = 1.52956262E-02;
    COFD[3788] = -2.15278182E+01;
    COFD[3789] = 4.96219227E+00;
    COFD[3790] = -3.70270843E-01;
    COFD[3791] = 1.37072211E-02;
    COFD[3792] = -2.22638165E+01;
    COFD[3793] = 5.58490856E+00;
    COFD[3794] = -4.81588720E-01;
    COFD[3795] = 1.97445317E-02;
    COFD[3796] = -2.25004333E+01;
    COFD[3797] = 5.52198915E+00;
    COFD[3798] = -4.67014474E-01;
    COFD[3799] = 1.88574253E-02;
    COFD[3800] = -2.21822461E+01;
    COFD[3801] = 5.60465338E+00;
    COFD[3802] = -4.88572478E-01;
    COFD[3803] = 2.02248525E-02;
    COFD[3804] = -2.19389389E+01;
    COFD[3805] = 5.15446948E+00;
    COFD[3806] = -4.01332769E-01;
    COFD[3807] = 1.52956262E-02;
    COFD[3808] = -2.25140525E+01;
    COFD[3809] = 5.49776513E+00;
    COFD[3810] = -4.61463030E-01;
    COFD[3811] = 1.85209236E-02;
    COFD[3812] = -2.24969995E+01;
    COFD[3813] = 5.50509817E+00;
    COFD[3814] = -4.63572794E-01;
    COFD[3815] = 1.86581046E-02;
    COFD[3816] = -2.19508859E+01;
    COFD[3817] = 4.96368178E+00;
    COFD[3818] = -3.70505465E-01;
    COFD[3819] = 1.37190339E-02;
    COFD[3820] = -2.25760230E+01;
    COFD[3821] = 5.41049872E+00;
    COFD[3822] = -4.45356411E-01;
    COFD[3823] = 1.76320470E-02;
    COFD[3824] = -2.25717119E+01;
    COFD[3825] = 5.41536807E+00;
    COFD[3826] = -4.46269562E-01;
    COFD[3827] = 1.76828228E-02;
    COFD[3828] = -2.25762645E+01;
    COFD[3829] = 5.41536807E+00;
    COFD[3830] = -4.46269562E-01;
    COFD[3831] = 1.76828228E-02;
    COFD[3832] = -2.24342646E+01;
    COFD[3833] = 5.06106414E+00;
    COFD[3834] = -3.86053039E-01;
    COFD[3835] = 1.45081784E-02;
    COFD[3836] = -2.24360850E+01;
    COFD[3837] = 5.06106414E+00;
    COFD[3838] = -3.86053039E-01;
    COFD[3839] = 1.45081784E-02;
    COFD[3840] = -2.10674485E+01;
    COFD[3841] = 5.15027524E+00;
    COFD[3842] = -4.46126111E-01;
    COFD[3843] = 1.90401391E-02;
    COFD[3844] = -1.99785176E+01;
    COFD[3845] = 4.92184026E+00;
    COFD[3846] = -4.19745472E-01;
    COFD[3847] = 1.80268154E-02;
    COFD[3848] = -1.64898528E+01;
    COFD[3849] = 4.01175649E+00;
    COFD[3850] = -3.08860971E-01;
    COFD[3851] = 1.35100076E-02;
    COFD[3852] = -2.03113704E+01;
    COFD[3853] = 5.50136606E+00;
    COFD[3854] = -4.82461887E-01;
    COFD[3855] = 2.02471523E-02;
    COFD[3856] = -2.00047095E+01;
    COFD[3857] = 4.92184026E+00;
    COFD[3858] = -4.19745472E-01;
    COFD[3859] = 1.80268154E-02;
    COFD[3860] = -1.91326792E+01;
    COFD[3861] = 3.82263611E+00;
    COFD[3862] = -1.93983472E-01;
    COFD[3863] = 4.95789388E-03;
    COFD[3864] = -2.13955999E+01;
    COFD[3865] = 5.25183817E+00;
    COFD[3866] = -4.57376333E-01;
    COFD[3867] = 1.94504429E-02;
    COFD[3868] = -2.14072803E+01;
    COFD[3869] = 5.25183817E+00;
    COFD[3870] = -4.57376333E-01;
    COFD[3871] = 1.94504429E-02;
    COFD[3872] = -2.14185232E+01;
    COFD[3873] = 5.25183817E+00;
    COFD[3874] = -4.57376333E-01;
    COFD[3875] = 1.94504429E-02;
    COFD[3876] = -2.28655752E+01;
    COFD[3877] = 5.50522401E+00;
    COFD[3878] = -4.63604304E-01;
    COFD[3879] = 1.86600785E-02;
    COFD[3880] = -2.10844012E+01;
    COFD[3881] = 5.15315713E+00;
    COFD[3882] = -4.46344043E-01;
    COFD[3883] = 1.90431546E-02;
    COFD[3884] = -2.04620510E+01;
    COFD[3885] = 4.26473557E+00;
    COFD[3886] = -2.61033037E-01;
    COFD[3887] = 8.23906412E-03;
    COFD[3888] = -2.19248250E+01;
    COFD[3889] = 5.49350509E+00;
    COFD[3890] = -4.81613405E-01;
    COFD[3891] = 2.02171734E-02;
    COFD[3892] = -2.19550907E+01;
    COFD[3893] = 5.49350509E+00;
    COFD[3894] = -4.81613405E-01;
    COFD[3895] = 2.02171734E-02;
    COFD[3896] = -2.19053841E+01;
    COFD[3897] = 5.47162499E+00;
    COFD[3898] = -4.79195552E-01;
    COFD[3899] = 2.01289088E-02;
    COFD[3900] = -2.06858147E+01;
    COFD[3901] = 4.35920123E+00;
    COFD[3902] = -2.75491273E-01;
    COFD[3903] = 8.95100289E-03;
    COFD[3904] = -2.28446667E+01;
    COFD[3905] = 5.50134401E+00;
    COFD[3906] = -4.62488197E-01;
    COFD[3907] = 1.85873697E-02;
    COFD[3908] = -2.13524540E+01;
    COFD[3909] = 4.61201872E+00;
    COFD[3910] = -3.14803338E-01;
    COFD[3911] = 1.09082984E-02;
    COFD[3912] = -2.13838498E+01;
    COFD[3913] = 4.61201872E+00;
    COFD[3914] = -3.14803338E-01;
    COFD[3915] = 1.09082984E-02;
    COFD[3916] = -2.08236367E+01;
    COFD[3917] = 4.35920123E+00;
    COFD[3918] = -2.75491273E-01;
    COFD[3919] = 8.95100289E-03;
    COFD[3920] = -2.26089431E+01;
    COFD[3921] = 5.44867280E+00;
    COFD[3922] = -4.52284883E-01;
    COFD[3923] = 1.80110706E-02;
    COFD[3924] = -2.26029886E+01;
    COFD[3925] = 5.27383847E+00;
    COFD[3926] = -4.21722368E-01;
    COFD[3927] = 1.63729618E-02;
    COFD[3928] = -2.26579938E+01;
    COFD[3929] = 5.52001624E+00;
    COFD[3930] = -4.66629503E-01;
    COFD[3931] = 1.88355817E-02;
    COFD[3932] = -2.13607457E+01;
    COFD[3933] = 4.61201872E+00;
    COFD[3934] = -3.14803338E-01;
    COFD[3935] = 1.09082984E-02;
    COFD[3936] = -2.25160816E+01;
    COFD[3937] = 5.21003123E+00;
    COFD[3938] = -4.10612564E-01;
    COFD[3939] = 1.57798598E-02;
    COFD[3940] = -2.25635595E+01;
    COFD[3941] = 5.24330646E+00;
    COFD[3942] = -4.16370120E-01;
    COFD[3943] = 1.60860486E-02;
    COFD[3944] = -2.12332312E+01;
    COFD[3945] = 4.36095377E+00;
    COFD[3946] = -2.75760539E-01;
    COFD[3947] = 8.96430249E-03;
    COFD[3948] = -2.24161979E+01;
    COFD[3949] = 5.05061421E+00;
    COFD[3950] = -3.84359196E-01;
    COFD[3951] = 1.44214004E-02;
    COFD[3952] = -2.24284563E+01;
    COFD[3953] = 5.06106414E+00;
    COFD[3954] = -3.86053039E-01;
    COFD[3955] = 1.45081784E-02;
    COFD[3956] = -2.24342646E+01;
    COFD[3957] = 5.06106414E+00;
    COFD[3958] = -3.86053039E-01;
    COFD[3959] = 1.45081784E-02;
    COFD[3960] = -2.17746534E+01;
    COFD[3961] = 4.48837319E+00;
    COFD[3962] = -2.95423315E-01;
    COFD[3963] = 9.93861345E-03;
    COFD[3964] = -2.17771745E+01;
    COFD[3965] = 4.48837319E+00;
    COFD[3966] = -2.95423315E-01;
    COFD[3967] = 9.93861345E-03;
    COFD[3968] = -2.10685573E+01;
    COFD[3969] = 5.15027524E+00;
    COFD[3970] = -4.46126111E-01;
    COFD[3971] = 1.90401391E-02;
    COFD[3972] = -1.99792167E+01;
    COFD[3973] = 4.92184026E+00;
    COFD[3974] = -4.19745472E-01;
    COFD[3975] = 1.80268154E-02;
    COFD[3976] = -1.64899530E+01;
    COFD[3977] = 4.01175649E+00;
    COFD[3978] = -3.08860971E-01;
    COFD[3979] = 1.35100076E-02;
    COFD[3980] = -2.03114210E+01;
    COFD[3981] = 5.50136606E+00;
    COFD[3982] = -4.82461887E-01;
    COFD[3983] = 2.02471523E-02;
    COFD[3984] = -2.00054461E+01;
    COFD[3985] = 4.92184026E+00;
    COFD[3986] = -4.19745472E-01;
    COFD[3987] = 1.80268154E-02;
    COFD[3988] = -1.91334529E+01;
    COFD[3989] = 3.82263611E+00;
    COFD[3990] = -1.93983472E-01;
    COFD[3991] = 4.95789388E-03;
    COFD[3992] = -2.13968281E+01;
    COFD[3993] = 5.25183817E+00;
    COFD[3994] = -4.57376333E-01;
    COFD[3995] = 1.94504429E-02;
    COFD[3996] = -2.14085375E+01;
    COFD[3997] = 5.25183817E+00;
    COFD[3998] = -4.57376333E-01;
    COFD[3999] = 1.94504429E-02;
    COFD[4000] = -2.14198091E+01;
    COFD[4001] = 5.25183817E+00;
    COFD[4002] = -4.57376333E-01;
    COFD[4003] = 1.94504429E-02;
    COFD[4004] = -2.28671232E+01;
    COFD[4005] = 5.50522401E+00;
    COFD[4006] = -4.63604304E-01;
    COFD[4007] = 1.86600785E-02;
    COFD[4008] = -2.10855099E+01;
    COFD[4009] = 5.15315713E+00;
    COFD[4010] = -4.46344043E-01;
    COFD[4011] = 1.90431546E-02;
    COFD[4012] = -2.04632210E+01;
    COFD[4013] = 4.26473557E+00;
    COFD[4014] = -2.61033037E-01;
    COFD[4015] = 8.23906412E-03;
    COFD[4016] = -2.19254485E+01;
    COFD[4017] = 5.49350509E+00;
    COFD[4018] = -4.81613405E-01;
    COFD[4019] = 2.02171734E-02;
    COFD[4020] = -2.19557531E+01;
    COFD[4021] = 5.49350509E+00;
    COFD[4022] = -4.81613405E-01;
    COFD[4023] = 2.02171734E-02;
    COFD[4024] = -2.19060847E+01;
    COFD[4025] = 5.47162499E+00;
    COFD[4026] = -4.79195552E-01;
    COFD[4027] = 2.01289088E-02;
    COFD[4028] = -2.06870442E+01;
    COFD[4029] = 4.35920123E+00;
    COFD[4030] = -2.75491273E-01;
    COFD[4031] = 8.95100289E-03;
    COFD[4032] = -2.28458380E+01;
    COFD[4033] = 5.50134401E+00;
    COFD[4034] = -4.62488197E-01;
    COFD[4035] = 1.85873697E-02;
    COFD[4036] = -2.13539532E+01;
    COFD[4037] = 4.61201872E+00;
    COFD[4038] = -3.14803338E-01;
    COFD[4039] = 1.09082984E-02;
    COFD[4040] = -2.13854464E+01;
    COFD[4041] = 4.61201872E+00;
    COFD[4042] = -3.14803338E-01;
    COFD[4043] = 1.09082984E-02;
    COFD[4044] = -2.08252570E+01;
    COFD[4045] = 4.35920123E+00;
    COFD[4046] = -2.75491273E-01;
    COFD[4047] = 8.95100289E-03;
    COFD[4048] = -2.26099899E+01;
    COFD[4049] = 5.44867280E+00;
    COFD[4050] = -4.52284883E-01;
    COFD[4051] = 1.80110706E-02;
    COFD[4052] = -2.26044889E+01;
    COFD[4053] = 5.27383847E+00;
    COFD[4054] = -4.21722368E-01;
    COFD[4055] = 1.63729618E-02;
    COFD[4056] = -2.26591038E+01;
    COFD[4057] = 5.52001624E+00;
    COFD[4058] = -4.66629503E-01;
    COFD[4059] = 1.88355817E-02;
    COFD[4060] = -2.13622700E+01;
    COFD[4061] = 4.61201872E+00;
    COFD[4062] = -3.14803338E-01;
    COFD[4063] = 1.09082984E-02;
    COFD[4064] = -2.25175307E+01;
    COFD[4065] = 5.21003123E+00;
    COFD[4066] = -4.10612564E-01;
    COFD[4067] = 1.57798598E-02;
    COFD[4068] = -2.25650343E+01;
    COFD[4069] = 5.24330646E+00;
    COFD[4070] = -4.16370120E-01;
    COFD[4071] = 1.60860486E-02;
    COFD[4072] = -2.12354028E+01;
    COFD[4073] = 4.36095377E+00;
    COFD[4074] = -2.75760539E-01;
    COFD[4075] = 8.96430249E-03;
    COFD[4076] = -2.24179759E+01;
    COFD[4077] = 5.05061421E+00;
    COFD[4078] = -3.84359196E-01;
    COFD[4079] = 1.44214004E-02;
    COFD[4080] = -2.24302556E+01;
    COFD[4081] = 5.06106414E+00;
    COFD[4082] = -3.86053039E-01;
    COFD[4083] = 1.45081784E-02;
    COFD[4084] = -2.24360850E+01;
    COFD[4085] = 5.06106414E+00;
    COFD[4086] = -3.86053039E-01;
    COFD[4087] = 1.45081784E-02;
    COFD[4088] = -2.17771745E+01;
    COFD[4089] = 4.48837319E+00;
    COFD[4090] = -2.95423315E-01;
    COFD[4091] = 9.93861345E-03;
    COFD[4092] = -2.17797084E+01;
    COFD[4093] = 4.48837319E+00;
    COFD[4094] = -2.95423315E-01;
    COFD[4095] = 9.93861345E-03;
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
    COFTD[48] = 3.24747031E-01;
    COFTD[49] = 1.77798548E-04;
    COFTD[50] = -1.08934732E-07;
    COFTD[51] = 2.03595881E-11;
    COFTD[52] = 3.31191185E-01;
    COFTD[53] = 1.81326714E-04;
    COFTD[54] = -1.11096391E-07;
    COFTD[55] = 2.07635959E-11;
    COFTD[56] = 3.39557243E-01;
    COFTD[57] = 1.79335036E-04;
    COFTD[58] = -1.10135705E-07;
    COFTD[59] = 2.06427239E-11;
    COFTD[60] = 1.31424053E-01;
    COFTD[61] = 6.16429134E-04;
    COFTD[62] = -3.28571348E-07;
    COFTD[63] = 5.46153434E-11;
    COFTD[64] = 2.78021896E-01;
    COFTD[65] = 3.89608886E-04;
    COFTD[66] = -2.23546590E-07;
    COFTD[67] = 3.92078724E-11;
    COFTD[68] = 1.59288984E-01;
    COFTD[69] = 6.02833801E-04;
    COFTD[70] = -3.24837576E-07;
    COFTD[71] = 5.43909010E-11;
    COFTD[72] = 1.60621157E-01;
    COFTD[73] = 6.07875449E-04;
    COFTD[74] = -3.27554273E-07;
    COFTD[75] = 5.48457855E-11;
    COFTD[76] = 1.36817715E-01;
    COFTD[77] = 6.41727473E-04;
    COFTD[78] = -3.42055963E-07;
    COFTD[79] = 5.68567648E-11;
    COFTD[80] = 2.58066832E-01;
    COFTD[81] = 4.05072593E-04;
    COFTD[82] = -2.30587443E-07;
    COFTD[83] = 4.01863841E-11;
    COFTD[84] = 2.40639006E-01;
    COFTD[85] = 4.82930111E-04;
    COFTD[86] = -2.70362190E-07;
    COFTD[87] = 4.65173265E-11;
    COFTD[88] = 2.82974392E-01;
    COFTD[89] = 3.73032949E-04;
    COFTD[90] = -2.14959161E-07;
    COFTD[91] = 3.78355155E-11;
    COFTD[92] = 1.59647939E-01;
    COFTD[93] = 6.04192274E-04;
    COFTD[94] = -3.25569591E-07;
    COFTD[95] = 5.45134698E-11;
    COFTD[96] = 2.27261590E-01;
    COFTD[97] = 4.99550076E-04;
    COFTD[98] = -2.78004320E-07;
    COFTD[99] = 4.76209155E-11;
    COFTD[100] = 2.34098762E-01;
    COFTD[101] = 4.91099181E-04;
    COFTD[102] = -2.74133967E-07;
    COFTD[103] = 4.70636702E-11;
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
    COFTD[120] = 1.55057696E-01;
    COFTD[121] = 6.55342553E-04;
    COFTD[122] = -3.51111538E-07;
    COFTD[123] = 5.85627121E-11;
    COFTD[124] = 1.55121130E-01;
    COFTD[125] = 6.55610653E-04;
    COFTD[126] = -3.51255177E-07;
    COFTD[127] = 5.85866700E-11;
    COFTD[128] = 2.01521643E-01;
    COFTD[129] = 5.62744089E-04;
    COFTD[130] = -3.08519239E-07;
    COFTD[131] = 5.22805986E-11;
    COFTD[132] = 2.35283119E-01;
    COFTD[133] = 4.65670599E-04;
    COFTD[134] = -2.60939824E-07;
    COFTD[135] = 4.49271822E-11;
    COFTD[136] = 1.44152190E-01;
    COFTD[137] = 7.99993584E-05;
    COFTD[138] = -4.89707442E-08;
    COFTD[139] = 9.14277269E-12;
    COFTD[140] = 0.00000000E+00;
    COFTD[141] = 0.00000000E+00;
    COFTD[142] = 0.00000000E+00;
    COFTD[143] = 0.00000000E+00;
    COFTD[144] = 2.37053352E-01;
    COFTD[145] = 4.69174231E-04;
    COFTD[146] = -2.62903094E-07;
    COFTD[147] = 4.52652072E-11;
    COFTD[148] = -1.74352698E-01;
    COFTD[149] = 8.62246873E-04;
    COFTD[150] = -3.79545489E-07;
    COFTD[151] = 5.60262093E-11;
    COFTD[152] = 1.79840299E-01;
    COFTD[153] = 6.01722902E-04;
    COFTD[154] = -3.26433894E-07;
    COFTD[155] = 5.49112302E-11;
    COFTD[156] = 1.80186965E-01;
    COFTD[157] = 6.02882805E-04;
    COFTD[158] = -3.27063140E-07;
    COFTD[159] = 5.50170790E-11;
    COFTD[160] = 1.80513677E-01;
    COFTD[161] = 6.03975942E-04;
    COFTD[162] = -3.27656165E-07;
    COFTD[163] = 5.51168351E-11;
    COFTD[164] = -2.00309448E-02;
    COFTD[165] = 8.50440115E-04;
    COFTD[166] = -4.21064468E-07;
    COFTD[167] = 6.67959710E-11;
    COFTD[168] = 2.00119897E-01;
    COFTD[169] = 5.64793704E-04;
    COFTD[170] = -3.09445484E-07;
    COFTD[171] = 5.24139335E-11;
    COFTD[172] = -1.61357564E-01;
    COFTD[173] = 9.05920260E-04;
    COFTD[174] = -4.07879153E-07;
    COFTD[175] = 6.10626290E-11;
    COFTD[176] = 9.90752318E-02;
    COFTD[177] = 6.44201384E-04;
    COFTD[178] = -3.38485953E-07;
    COFTD[179] = 5.57356746E-11;
    COFTD[180] = 1.00039110E-01;
    COFTD[181] = 6.50468660E-04;
    COFTD[182] = -3.41778999E-07;
    COFTD[183] = 5.62779132E-11;
    COFTD[184] = 1.05124122E-01;
    COFTD[185] = 6.50665957E-04;
    COFTD[186] = -3.42564538E-07;
    COFTD[187] = 5.64804120E-11;
    COFTD[188] = -1.56651581E-01;
    COFTD[189] = 9.09789751E-04;
    COFTD[190] = -4.11714242E-07;
    COFTD[191] = 6.18310893E-11;
    COFTD[192] = -2.28637575E-02;
    COFTD[193] = 8.35412914E-04;
    COFTD[194] = -4.12929260E-07;
    COFTD[195] = 6.54380945E-11;
    COFTD[196] = -1.41640506E-01;
    COFTD[197] = 9.21404324E-04;
    COFTD[198] = -4.23210110E-07;
    COFTD[199] = 6.41400322E-11;
    COFTD[200] = -1.42230624E-01;
    COFTD[201] = 9.25243177E-04;
    COFTD[202] = -4.24973333E-07;
    COFTD[203] = 6.44072593E-11;
    COFTD[204] = -1.59826932E-01;
    COFTD[205] = 9.28231324E-04;
    COFTD[206] = -4.20059750E-07;
    COFTD[207] = 6.30844146E-11;
    COFTD[208] = -3.81470765E-02;
    COFTD[209] = 8.39833490E-04;
    COFTD[210] = -4.11688915E-07;
    COFTD[211] = 6.49124952E-11;
    COFTD[212] = -7.23038994E-02;
    COFTD[213] = 8.89466098E-04;
    COFTD[214] = -4.28124818E-07;
    COFTD[215] = 6.67586244E-11;
    COFTD[216] = -1.42100396E-02;
    COFTD[217] = 8.23812102E-04;
    COFTD[218] = -4.08995515E-07;
    COFTD[219] = 6.49899310E-11;
    COFTD[220] = -1.41799739E-01;
    COFTD[221] = 9.22440172E-04;
    COFTD[222] = -4.23685885E-07;
    COFTD[223] = 6.42121388E-11;
    COFTD[224] = -8.35962674E-02;
    COFTD[225] = 8.94627716E-04;
    COFTD[226] = -4.27681210E-07;
    COFTD[227] = 6.64140378E-11;
    COFTD[228] = -7.78657454E-02;
    COFTD[229] = 8.92101015E-04;
    COFTD[230] = -4.27969255E-07;
    COFTD[231] = 6.66000503E-11;
    COFTD[232] = -1.62301128E-01;
    COFTD[233] = 9.43217155E-04;
    COFTD[234] = -4.26881994E-07;
    COFTD[235] = 6.41127358E-11;
    COFTD[236] = -1.04463705E-01;
    COFTD[237] = 9.17317898E-04;
    COFTD[238] = -4.33159478E-07;
    COFTD[239] = 6.67640055E-11;
    COFTD[240] = -1.03383911E-01;
    COFTD[241] = 9.17368946E-04;
    COFTD[242] = -4.33506203E-07;
    COFTD[243] = 6.68476435E-11;
    COFTD[244] = -1.03451906E-01;
    COFTD[245] = 9.17972300E-04;
    COFTD[246] = -4.33791320E-07;
    COFTD[247] = 6.68916093E-11;
    COFTD[248] = -1.55196478E-01;
    COFTD[249] = 9.48793311E-04;
    COFTD[250] = -4.32429136E-07;
    COFTD[251] = 6.52268964E-11;
    COFTD[252] = -1.55228210E-01;
    COFTD[253] = 9.48987307E-04;
    COFTD[254] = -4.32517553E-07;
    COFTD[255] = 6.52402330E-11;
}

/* Replace this routine with the one generated by the Gauss Jordan solver of DW */
AMREX_GPU_HOST_DEVICE void sgjsolve(double* A, double* x, double* b) {
    amrex::Abort("sgjsolve not implemented, choose a different solver ");
}

/* Replace this routine with the one generated by the Gauss Jordan solver of DW */
AMREX_GPU_HOST_DEVICE void sgjsolve_simplified(double* A, double* x, double* b) {
    amrex::Abort("sgjsolve_simplified not implemented, choose a different solver ");
}

