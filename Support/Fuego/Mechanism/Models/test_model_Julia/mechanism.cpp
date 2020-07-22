#include "chemistry_file.H"

#ifndef AMREX_USE_CUDA
namespace thermo
{
    double fwd_A[10], fwd_beta[10], fwd_Ea[10];
    double low_A[10], low_beta[10], low_Ea[10];
    double rev_A[10], rev_beta[10], rev_Ea[10];
    double troe_a[10],troe_Ts[10], troe_Tss[10], troe_Tsss[10];
    double sri_a[10], sri_b[10], sri_c[10], sri_d[10], sri_e[10];
    double activation_units[10], prefactor_units[10], phase_units[10];
    int is_PD[10], troe_len[10], sri_len[10], nTB[10], *TBid[10];
    double *TB[10];
    std::vector<std::vector<double>> kiv(10); 
    std::vector<std::vector<double>> nuv(10); 
};

using namespace thermo;
#endif

/* Inverse molecular weights */
static AMREX_GPU_DEVICE_MANAGED double imw[6] = {
    1.0 / 2.015940,  /*H2 */
    1.0 / 1.007970,  /*H */
    1.0 / 15.999400,  /*O */
    1.0 / 17.007370,  /*OH */
    1.0 / 44.009950,  /*CO2 */
    1.0 / 29.018520};  /*HCO */

/* Inverse molecular weights */
static AMREX_GPU_DEVICE_MANAGED double molecular_weights[6] = {
    2.015940,  /*H2 */
    1.007970,  /*H */
    15.999400,  /*O */
    17.007370,  /*OH */
    44.009950,  /*CO2 */
    29.018520};  /*HCO */

AMREX_GPU_HOST_DEVICE
void get_imw(double imw_new[]){
    for(int i = 0; i<6; ++i) imw_new[i] = imw[i];
}

/* TODO: check necessity because redundant with CKWT */
AMREX_GPU_HOST_DEVICE
void get_mw(double mw_new[]){
    for(int i = 0; i<6; ++i) mw_new[i] = molecular_weights[i];
}


#ifndef AMREX_USE_CUDA
/* Initializes parameter database */
void CKINIT()
{

    // (0):  O + HO2 => OH + O2
    kiv[0] = {2,3};
    nuv[0] = {-1,1};
    kiv_qss[0] = {2,0};
    nuv_qss[0] = {-1,1};
    // (0):  O + HO2 => OH + O2
    fwd_A[0]     = 20000000000000;
    fwd_beta[0]  = 0;
    fwd_Ea[0]    = 0;
    prefactor_units[0]  = 1.0000000000000002e-06;
    activation_units[0] = 0.50321666580471969;
    phase_units[0]      = pow(10,-12.000000);
    is_PD[0] = 0;
    nTB[0] = 0;

    // (1):  H + HO2 => O + H2O
    kiv[1] = {1,2};
    nuv[1] = {-1,1};
    kiv_qss[1] = {2,1};
    nuv_qss[1] = {-1,1};
    // (1):  H + HO2 => O + H2O
    fwd_A[1]     = 3970000000000;
    fwd_beta[1]  = 0;
    fwd_Ea[1]    = 671;
    prefactor_units[1]  = 1.0000000000000002e-06;
    activation_units[1] = 0.50321666580471969;
    phase_units[1]      = pow(10,-12.000000);
    is_PD[1] = 0;
    nTB[1] = 0;

    // (2):  H + H2O2 <=> OH + H2O
    kiv[2] = {1,3};
    nuv[2] = {-1,1};
    kiv_qss[2] = {3,1};
    nuv_qss[2] = {-1,1};
    // (2):  H + H2O2 <=> OH + H2O
    fwd_A[2]     = 10000000000000;
    fwd_beta[2]  = 0;
    fwd_Ea[2]    = 3600;
    prefactor_units[2]  = 1.0000000000000002e-06;
    activation_units[2] = 0.50321666580471969;
    phase_units[2]      = pow(10,-12.000000);
    is_PD[2] = 0;
    nTB[2] = 0;

    // (3):  O + CH => H + CO
    kiv[3] = {2,1};
    nuv[3] = {-1,1};
    kiv_qss[3] = {5,7};
    nuv_qss[3] = {-1,1};
    // (3):  O + CH => H + CO
    fwd_A[3]     = 57000000000000;
    fwd_beta[3]  = 0;
    fwd_Ea[3]    = 0;
    prefactor_units[3]  = 1.0000000000000002e-06;
    activation_units[3] = 0.50321666580471969;
    phase_units[3]      = pow(10,-12.000000);
    is_PD[3] = 0;
    nTB[3] = 0;

    // (4):  H + CH <=> C + H2
    kiv[4] = {1,0};
    nuv[4] = {-1,1};
    kiv_qss[4] = {5,4};
    nuv_qss[4] = {-1,1};
    // (4):  H + CH <=> C + H2
    fwd_A[4]     = 110000000000000;
    fwd_beta[4]  = 0;
    fwd_Ea[4]    = 0;
    prefactor_units[4]  = 1.0000000000000002e-06;
    activation_units[4] = 0.50321666580471969;
    phase_units[4]      = pow(10,-12.000000);
    is_PD[4] = 0;
    nTB[4] = 0;

    // (5):  O + CH2 <=> H + HCO
    kiv[5] = {2,1,5};
    nuv[5] = {-1,1,1};
    kiv_qss[5] = {6};
    nuv_qss[5] = {-1};
    // (5):  O + CH2 <=> H + HCO
    fwd_A[5]     = 80000000000000;
    fwd_beta[5]  = 0;
    fwd_Ea[5]    = 0;
    prefactor_units[5]  = 1.0000000000000002e-06;
    activation_units[5] = 0.50321666580471969;
    phase_units[5]      = pow(10,-12.000000);
    is_PD[5] = 0;
    nTB[5] = 0;

    // (6):  H + O2 <=> O + OH
    kiv[6] = {1,2,3};
    nuv[6] = {-1,1,1};
    kiv_qss[6] = {0};
    nuv_qss[6] = {-1};
    // (6):  H + O2 <=> O + OH
    fwd_A[6]     = 83000000000000;
    fwd_beta[6]  = 0;
    fwd_Ea[6]    = 14413;
    prefactor_units[6]  = 1.0000000000000002e-06;
    activation_units[6] = 0.50321666580471969;
    phase_units[6]      = pow(10,-12.000000);
    is_PD[6] = 0;
    nTB[6] = 0;

    // (7):  H + HO2 <=> 2.000000 OH
    kiv[7] = {1,3};
    nuv[7] = {-1,2.0};
    kiv_qss[7] = {2};
    nuv_qss[7] = {-1};
    // (7):  H + HO2 <=> 2.000000 OH
    fwd_A[7]     = 134000000000000;
    fwd_beta[7]  = 0;
    fwd_Ea[7]    = 635;
    prefactor_units[7]  = 1.0000000000000002e-06;
    activation_units[7] = 0.50321666580471969;
    phase_units[7]      = pow(10,-12.000000);
    is_PD[7] = 0;
    nTB[7] = 0;

    // (8):  OH + CO <=> H + CO2
    kiv[8] = {3,1,4};
    nuv[8] = {-1,1,1};
    kiv_qss[8] = {7};
    nuv_qss[8] = {-1};
    // (8):  OH + CO <=> H + CO2
    fwd_A[8]     = 47600000;
    fwd_beta[8]  = 1.228;
    fwd_Ea[8]    = 70;
    prefactor_units[8]  = 1.0000000000000002e-06;
    activation_units[8] = 0.50321666580471969;
    phase_units[8]      = pow(10,-12.000000);
    is_PD[8] = 0;
    nTB[8] = 0;

    // (9):  OH + CH <=> H + HCO
    kiv[9] = {3,1,5};
    nuv[9] = {-1,1,1};
    kiv_qss[9] = {5};
    nuv_qss[9] = {-1};
    // (9):  OH + CH <=> H + HCO
    fwd_A[9]     = 30000000000000;
    fwd_beta[9]  = 0;
    fwd_Ea[9]    = 0;
    prefactor_units[9]  = 1.0000000000000002e-06;
    activation_units[9] = 0.50321666580471969;
    phase_units[9]      = pow(10,-12.000000);
    is_PD[9] = 0;
    nTB[9] = 0;

}


/* Finalizes parameter database */
void CKFINALIZE()
{
  for (int i=0; i<10; ++i) {
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
    *mm = 3;
    *kk = 6;
    *ii = 10;
    *nfit = -1; /*Why do you need this anyway ?  */
}


/* Returns the vector of strings of element names */
void CKSYME_STR(amrex::Vector<std::string>& ename)
{
    ename.resize(3);
    ename[0] = "O";
    ename[1] = "H";
    ename[2] = "C";
}


/* Returns the vector of strings of species names */
void CKSYMS_STR(amrex::Vector<std::string>& kname)
{
    kname.resize(6);
    kname[0] = "H2";
    kname[1] = "H";
    kname[2] = "O";
    kname[3] = "OH";
    kname[4] = "CO2";
    kname[5] = "HCO";
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
    XW += x[0]*2.015940; /*H2 */
    XW += x[1]*1.007970; /*H */
    XW += x[2]*15.999400; /*O */
    XW += x[3]*17.007370; /*OH */
    XW += x[4]*44.009950; /*CO2 */
    XW += x[5]*29.018520; /*HCO */
    *P = *rho * 8.31446e+07 * (*T) / XW; /*P = rho*R*T/W */

    return;
}


/*Compute P = rhoRT/W(y) */
AMREX_GPU_HOST_DEVICE void CKPY(double *  rho, double *  T, double *  y,  double *  P)
{
    double YOW = 0;/* for computing mean MW */
    YOW += y[0]*imw[0]; /*H2 */
    YOW += y[1]*imw[1]; /*H */
    YOW += y[2]*imw[2]; /*O */
    YOW += y[3]*imw[3]; /*OH */
    YOW += y[4]*imw[4]; /*CO2 */
    YOW += y[5]*imw[5]; /*HCO */
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
    W += c[0]*2.015940; /*H2 */
    W += c[1]*1.007970; /*H */
    W += c[2]*15.999400; /*O */
    W += c[3]*17.007370; /*OH */
    W += c[4]*44.009950; /*CO2 */
    W += c[5]*29.018520; /*HCO */

    for (id = 0; id < 6; ++id) {
        sumC += c[id];
    }
    *P = *rho * 8.31446e+07 * (*T) * sumC / W; /*P = rho*R*T/W */

    return;
}


/*Compute rho = PW(x)/RT */
void CKRHOX(double *  P, double *  T, double *  x,  double *  rho)
{
    double XW = 0;/* To hold mean molecular wt */
    XW += x[0]*2.015940; /*H2 */
    XW += x[1]*1.007970; /*H */
    XW += x[2]*15.999400; /*O */
    XW += x[3]*17.007370; /*OH */
    XW += x[4]*44.009950; /*CO2 */
    XW += x[5]*29.018520; /*HCO */
    *rho = *P * XW / (8.31446e+07 * (*T)); /*rho = P*W/(R*T) */

    return;
}


/*Compute rho = P*W(y)/RT */
AMREX_GPU_HOST_DEVICE void CKRHOY(double *  P, double *  T, double *  y,  double *  rho)
{
    double YOW = 0;
    double tmp[6];

    for (int i = 0; i < 6; i++)
    {
        tmp[i] = y[i]*imw[i];
    }
    for (int i = 0; i < 6; i++)
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
    W += c[0]*2.015940; /*H2 */
    W += c[1]*1.007970; /*H */
    W += c[2]*15.999400; /*O */
    W += c[3]*17.007370; /*OH */
    W += c[4]*44.009950; /*CO2 */
    W += c[5]*29.018520; /*HCO */

    for (id = 0; id < 6; ++id) {
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
    double tmp[6];

    for (int i = 0; i < 6; i++)
    {
        tmp[i] = y[i]*imw[i];
    }
    for (int i = 0; i < 6; i++)
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
    XW += x[0]*2.015940; /*H2 */
    XW += x[1]*1.007970; /*H */
    XW += x[2]*15.999400; /*O */
    XW += x[3]*17.007370; /*OH */
    XW += x[4]*44.009950; /*CO2 */
    XW += x[5]*29.018520; /*HCO */
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
    W += c[0]*2.015940; /*H2 */
    W += c[1]*1.007970; /*H */
    W += c[2]*15.999400; /*O */
    W += c[3]*17.007370; /*OH */
    W += c[4]*44.009950; /*CO2 */
    W += c[5]*29.018520; /*HCO */

    for (id = 0; id < 6; ++id) {
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
    double tmp[6];

    for (int i = 0; i < 6; i++)
    {
        tmp[i] = y[i]*imw[i];
    }
    for (int i = 0; i < 6; i++)
    {
        YOW += tmp[i];
    }

    double YOWINV = 1.0/YOW;

    for (int i = 0; i < 6; i++)
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
    for (int i = 0; i < 6; i++)
    {
        c[i] = y[i]*imw[i];
    }
    for (int i = 0; i < 6; i++)
    {
        YOW += c[i];
    }

    /*PW/RT (see Eq. 7) */
    PWORT = (*P)/(YOW * 8.31446e+07 * (*T)); 
    /*Now compute conversion */

    for (int i = 0; i < 6; i++)
    {
        c[i] = PWORT * y[i] * imw[i];
    }
    return;
}


/*convert y[species] (mass fracs) to c[species] (molar conc) */
AMREX_GPU_HOST_DEVICE void CKYTCR(double *  rho, double *  T, double *  y,  double *  c)
{
    for (int i = 0; i < 6; i++)
    {
        c[i] = (*rho)  * y[i] * imw[i];
    }
}


/*convert x[species] (mole fracs) to y[species] (mass fracs) */
AMREX_GPU_HOST_DEVICE void CKXTY(double *  x,  double *  y)
{
    double XW = 0; /*See Eq 4, 9 in CK Manual */
    /*Compute mean molecular wt first */
    XW += x[0]*2.015940; /*H2 */
    XW += x[1]*1.007970; /*H */
    XW += x[2]*15.999400; /*O */
    XW += x[3]*17.007370; /*OH */
    XW += x[4]*44.009950; /*CO2 */
    XW += x[5]*29.018520; /*HCO */
    /*Now compute conversion */
    double XWinv = 1.0/XW;
    y[0] = x[0]*2.015940*XWinv; 
    y[1] = x[1]*1.007970*XWinv; 
    y[2] = x[2]*15.999400*XWinv; 
    y[3] = x[3]*17.007370*XWinv; 
    y[4] = x[4]*44.009950*XWinv; 
    y[5] = x[5]*29.018520*XWinv; 

    return;
}


/*convert x[species] (mole fracs) to c[species] (molar conc) */
void CKXTCP(double *  P, double *  T, double *  x,  double *  c)
{
    int id; /*loop counter */
    double PORT = (*P)/(8.31446e+07 * (*T)); /*P/RT */

    /*Compute conversion, see Eq 10 */
    for (id = 0; id < 6; ++id) {
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
    XW += x[0]*2.015940; /*H2 */
    XW += x[1]*1.007970; /*H */
    XW += x[2]*15.999400; /*O */
    XW += x[3]*17.007370; /*OH */
    XW += x[4]*44.009950; /*CO2 */
    XW += x[5]*29.018520; /*HCO */
    ROW = (*rho) / XW;

    /*Compute conversion, see Eq 11 */
    for (id = 0; id < 6; ++id) {
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
    for (id = 0; id < 6; ++id) {
        sumC += c[id];
    }

    /* See Eq 13  */
    double sumCinv = 1.0/sumC;
    for (id = 0; id < 6; ++id) {
        x[id] = c[id]*sumCinv;
    }

    return;
}


/*convert c[species] (molar conc) to y[species] (mass fracs) */
void CKCTY(double *  c, double *  y)
{
    double CW = 0; /*See Eq 12 in CK Manual */
    /*compute denominator in eq 12 first */
    CW += c[0]*2.015940; /*H2 */
    CW += c[1]*1.007970; /*H */
    CW += c[2]*15.999400; /*O */
    CW += c[3]*17.007370; /*OH */
    CW += c[4]*44.009950; /*CO2 */
    CW += c[5]*29.018520; /*HCO */
    /*Now compute conversion */
    double CWinv = 1.0/CW;
    y[0] = c[0]*2.015940*CWinv; 
    y[1] = c[1]*1.007970*CWinv; 
    y[2] = c[2]*15.999400*CWinv; 
    y[3] = c[3]*17.007370*CWinv; 
    y[4] = c[4]*44.009950*CWinv; 
    y[5] = c[5]*29.018520*CWinv; 

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
    cvms[0] *= 4.124360158612479e+07; /*H2 */
    cvms[1] *= 8.248720317224957e+07; /*H */
    cvms[2] *= 5.196734013871295e+06; /*O */
    cvms[3] *= 4.888740950630956e+06; /*OH */
    cvms[4] *= 1.889223372931176e+06; /*CO2 */
    cvms[5] *= 2.865226282440744e+06; /*HCO */
}


/*Returns the specific heats at constant pressure */
/*in mass units (Eq. 26) */
AMREX_GPU_HOST_DEVICE void CKCPMS(double *  T,  double *  cpms)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    cp_R(cpms, tc);
    /*multiply by R/molecularweight */
    cpms[0] *= 4.124360158612479e+07; /*H2 */
    cpms[1] *= 8.248720317224957e+07; /*H */
    cpms[2] *= 5.196734013871295e+06; /*O */
    cpms[3] *= 4.888740950630956e+06; /*OH */
    cpms[4] *= 1.889223372931176e+06; /*CO2 */
    cpms[5] *= 2.865226282440744e+06; /*HCO */
}


/*Returns internal energy in mass units (Eq 30.) */
AMREX_GPU_HOST_DEVICE void CKUMS(double *  T,  double *  ums)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446e+07*tT; /*R*T */
    speciesInternalEnergy(ums, tc);
    for (int i = 0; i < 6; i++)
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
    for (int i = 0; i < 6; i++)
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
    for (int i = 0; i < 6; i++)
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
    for (int i = 0; i < 6; i++)
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
    sms[0] *= 4.124360158612479e+07; /*H2 */
    sms[1] *= 8.248720317224957e+07; /*H */
    sms[2] *= 5.196734013871295e+06; /*O */
    sms[3] *= 4.888740950630956e+06; /*OH */
    sms[4] *= 1.889223372931176e+06; /*CO2 */
    sms[5] *= 2.865226282440744e+06; /*HCO */
}


/*Returns the mean specific heat at CP (Eq. 33) */
void CKCPBL(double *  T, double *  x,  double *  cpbl)
{
    int id; /*loop counter */
    double result = 0; 
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double cpor[6]; /* temporary storage */
    cp_R(cpor, tc);

    /*perform dot product */
    for (id = 0; id < 6; ++id) {
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
    double cpor[6], tresult[6]; /* temporary storage */
    cp_R(cpor, tc);
    for (int i = 0; i < 6; i++)
    {
        tresult[i] = cpor[i]*y[i]*imw[i];

    }
    for (int i = 0; i < 6; i++)
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
    double cvor[6]; /* temporary storage */
    cv_R(cvor, tc);

    /*perform dot product */
    for (id = 0; id < 6; ++id) {
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
    double cvor[6]; /* temporary storage */
    cv_R(cvor, tc);
    /*multiply by y/molecularweight */
    result += cvor[0]*y[0]*imw[0]; /*H2 */
    result += cvor[1]*y[1]*imw[1]; /*H */
    result += cvor[2]*y[2]*imw[2]; /*O */
    result += cvor[3]*y[3]*imw[3]; /*OH */
    result += cvor[4]*y[4]*imw[4]; /*CO2 */
    result += cvor[5]*y[5]*imw[5]; /*HCO */

    *cvbs = result * 8.31446e+07;
}


/*Returns the mean enthalpy of the mixture in molar units */
void CKHBML(double *  T, double *  x,  double *  hbml)
{
    int id; /*loop counter */
    double result = 0; 
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double hml[6]; /* temporary storage */
    double RT = 8.31446e+07*tT; /*R*T */
    speciesEnthalpy(hml, tc);

    /*perform dot product */
    for (id = 0; id < 6; ++id) {
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
    double hml[6], tmp[6]; /* temporary storage */
    double RT = 8.31446e+07*tT; /*R*T */
    speciesEnthalpy(hml, tc);
    int id;
    for (id = 0; id < 6; ++id) {
        tmp[id] = y[id]*hml[id]*imw[id];
    }
    for (id = 0; id < 6; ++id) {
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
    double uml[6]; /* temporary energy array */
    double RT = 8.31446e+07*tT; /*R*T */
    speciesInternalEnergy(uml, tc);

    /*perform dot product */
    for (id = 0; id < 6; ++id) {
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
    double ums[6]; /* temporary energy array */
    double RT = 8.31446e+07*tT; /*R*T */
    speciesInternalEnergy(ums, tc);
    /*perform dot product + scaling by wt */
    result += y[0]*ums[0]*imw[0]; /*H2 */
    result += y[1]*ums[1]*imw[1]; /*H */
    result += y[2]*ums[2]*imw[2]; /*O */
    result += y[3]*ums[3]*imw[3]; /*OH */
    result += y[4]*ums[4]*imw[4]; /*CO2 */
    result += y[5]*ums[5]*imw[5]; /*HCO */

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
    double sor[6]; /* temporary storage */
    speciesEntropy(sor, tc);

    /*Compute Eq 42 */
    for (id = 0; id < 6; ++id) {
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
    double sor[6]; /* temporary storage */
    double x[6]; /* need a ytx conversion */
    double YOW = 0; /*See Eq 4, 6 in CK Manual */
    /*Compute inverse of mean molecular wt first */
    YOW += y[0]*imw[0]; /*H2 */
    YOW += y[1]*imw[1]; /*H */
    YOW += y[2]*imw[2]; /*O */
    YOW += y[3]*imw[3]; /*OH */
    YOW += y[4]*imw[4]; /*CO2 */
    YOW += y[5]*imw[5]; /*HCO */
    /*Now compute y to x conversion */
    x[0] = y[0]/(2.015940*YOW); 
    x[1] = y[1]/(1.007970*YOW); 
    x[2] = y[2]/(15.999400*YOW); 
    x[3] = y[3]/(17.007370*YOW); 
    x[4] = y[4]/(44.009950*YOW); 
    x[5] = y[5]/(29.018520*YOW); 
    speciesEntropy(sor, tc);
    /*Perform computation in Eq 42 and 43 */
    result += x[0]*(sor[0]-log((x[0]+1e-100))-logPratio);
    result += x[1]*(sor[1]-log((x[1]+1e-100))-logPratio);
    result += x[2]*(sor[2]-log((x[2]+1e-100))-logPratio);
    result += x[3]*(sor[3]-log((x[3]+1e-100))-logPratio);
    result += x[4]*(sor[4]-log((x[4]+1e-100))-logPratio);
    result += x[5]*(sor[5]-log((x[5]+1e-100))-logPratio);
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
    double gort[6]; /* temporary storage */
    /*Compute g/RT */
    gibbs(gort, tc);

    /*Compute Eq 44 */
    for (id = 0; id < 6; ++id) {
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
    double gort[6]; /* temporary storage */
    double x[6]; /* need a ytx conversion */
    double YOW = 0; /*To hold 1/molecularweight */
    /*Compute inverse of mean molecular wt first */
    YOW += y[0]*imw[0]; /*H2 */
    YOW += y[1]*imw[1]; /*H */
    YOW += y[2]*imw[2]; /*O */
    YOW += y[3]*imw[3]; /*OH */
    YOW += y[4]*imw[4]; /*CO2 */
    YOW += y[5]*imw[5]; /*HCO */
    /*Now compute y to x conversion */
    x[0] = y[0]/(2.015940*YOW); 
    x[1] = y[1]/(1.007970*YOW); 
    x[2] = y[2]/(15.999400*YOW); 
    x[3] = y[3]/(17.007370*YOW); 
    x[4] = y[4]/(44.009950*YOW); 
    x[5] = y[5]/(29.018520*YOW); 
    gibbs(gort, tc);
    /*Perform computation in Eq 44 */
    result += x[0]*(gort[0]+log((x[0]+1e-100))+logPratio);
    result += x[1]*(gort[1]+log((x[1]+1e-100))+logPratio);
    result += x[2]*(gort[2]+log((x[2]+1e-100))+logPratio);
    result += x[3]*(gort[3]+log((x[3]+1e-100))+logPratio);
    result += x[4]*(gort[4]+log((x[4]+1e-100))+logPratio);
    result += x[5]*(gort[5]+log((x[5]+1e-100))+logPratio);
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
    double aort[6]; /* temporary storage */
    /*Compute g/RT */
    helmholtz(aort, tc);

    /*Compute Eq 44 */
    for (id = 0; id < 6; ++id) {
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
    double aort[6]; /* temporary storage */
    double x[6]; /* need a ytx conversion */
    double YOW = 0; /*To hold 1/molecularweight */
    /*Compute inverse of mean molecular wt first */
    YOW += y[0]*imw[0]; /*H2 */
    YOW += y[1]*imw[1]; /*H */
    YOW += y[2]*imw[2]; /*O */
    YOW += y[3]*imw[3]; /*OH */
    YOW += y[4]*imw[4]; /*CO2 */
    YOW += y[5]*imw[5]; /*HCO */
    /*Now compute y to x conversion */
    x[0] = y[0]/(2.015940*YOW); 
    x[1] = y[1]/(1.007970*YOW); 
    x[2] = y[2]/(15.999400*YOW); 
    x[3] = y[3]/(17.007370*YOW); 
    x[4] = y[4]/(44.009950*YOW); 
    x[5] = y[5]/(29.018520*YOW); 
    helmholtz(aort, tc);
    /*Perform computation in Eq 44 */
    result += x[0]*(aort[0]+log((x[0]+1e-100))+logPratio);
    result += x[1]*(aort[1]+log((x[1]+1e-100))+logPratio);
    result += x[2]*(aort[2]+log((x[2]+1e-100))+logPratio);
    result += x[3]*(aort[3]+log((x[3]+1e-100))+logPratio);
    result += x[4]*(aort[4]+log((x[4]+1e-100))+logPratio);
    result += x[5]*(aort[5]+log((x[5]+1e-100))+logPratio);
    /*Scale by RT/W */
    *abms = result * RT * YOW;
}


/*compute the production rate for each species */
AMREX_GPU_HOST_DEVICE void CKWC(double *  T, double *  C,  double *  wdot)
{
    int id; /*loop counter */

    /*convert to SI */
    for (id = 0; id < 6; ++id) {
        C[id] *= 1.0e6;
    }

    /*convert to chemkin units */
    productionRate(wdot, C, *T);

    /*convert to chemkin units */
    for (id = 0; id < 6; ++id) {
        C[id] *= 1.0e-6;
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the molar production rate of species */
/*Given P, T, and mass fractions */
void CKWYP(double *  P, double *  T, double *  y,  double *  wdot)
{
    int id; /*loop counter */
    double c[6]; /*temporary storage */
    double YOW = 0; 
    double PWORT; 
    /*Compute inverse of mean molecular wt first */
    YOW += y[0]*imw[0]; /*H2 */
    YOW += y[1]*imw[1]; /*H */
    YOW += y[2]*imw[2]; /*O */
    YOW += y[3]*imw[3]; /*OH */
    YOW += y[4]*imw[4]; /*CO2 */
    YOW += y[5]*imw[5]; /*HCO */
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

    /*convert to chemkin units */
    productionRate(wdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 6; ++id) {
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the molar production rate of species */
/*Given P, T, and mole fractions */
void CKWXP(double *  P, double *  T, double *  x,  double *  wdot)
{
    int id; /*loop counter */
    double c[6]; /*temporary storage */
    double PORT = 1e6 * (*P)/(8.31446e+07 * (*T)); /*1e6 * P/RT so c goes to SI units */

    /*Compute conversion, see Eq 10 */
    for (id = 0; id < 6; ++id) {
        c[id] = x[id]*PORT;
    }

    /*convert to chemkin units */
    productionRate(wdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 6; ++id) {
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the molar production rate of species */
/*Given rho, T, and mass fractions */
AMREX_GPU_HOST_DEVICE void CKWYR(double *  rho, double *  T, double *  y,  double *  wdot)
{
    int id; /*loop counter */
    double c[6]; /*temporary storage */
    /*See Eq 8 with an extra 1e6 so c goes to SI */
    c[0] = 1e6 * (*rho) * y[0]*imw[0]; 
    c[1] = 1e6 * (*rho) * y[1]*imw[1]; 
    c[2] = 1e6 * (*rho) * y[2]*imw[2]; 
    c[3] = 1e6 * (*rho) * y[3]*imw[3]; 
    c[4] = 1e6 * (*rho) * y[4]*imw[4]; 
    c[5] = 1e6 * (*rho) * y[5]*imw[5]; 

    /*call productionRate */
    productionRate(wdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 6; ++id) {
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the molar production rate of species */
/*Given rho, T, and mole fractions */
void CKWXR(double *  rho, double *  T, double *  x,  double *  wdot)
{
    int id; /*loop counter */
    double c[6]; /*temporary storage */
    double XW = 0; /*See Eq 4, 11 in CK Manual */
    double ROW; 
    /*Compute mean molecular wt first */
    XW += x[0]*2.015940; /*H2 */
    XW += x[1]*1.007970; /*H */
    XW += x[2]*15.999400; /*O */
    XW += x[3]*17.007370; /*OH */
    XW += x[4]*44.009950; /*CO2 */
    XW += x[5]*29.018520; /*HCO */
    /*Extra 1e6 factor to take c to SI */
    ROW = 1e6*(*rho) / XW;

    /*Compute conversion, see Eq 11 */
    for (id = 0; id < 6; ++id) {
        c[id] = x[id]*ROW;
    }

    /*convert to chemkin units */
    productionRate(wdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 6; ++id) {
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the elemental composition  */
/*of the speciesi (mdim is num of elements) */
void CKNCF(int * ncf)
{
    int id; /*loop counter */
    int kd = 3; 
    /*Zero ncf */
    for (id = 0; id < kd * 6; ++ id) {
         ncf[id] = 0; 
    }

    /*H2 */
    ncf[ 0 * kd + 1 ] = 2; /*H */

    /*H */
    ncf[ 1 * kd + 1 ] = 1; /*H */

    /*O */
    ncf[ 2 * kd + 0 ] = 1; /*O */

    /*OH */
    ncf[ 3 * kd + 0 ] = 1; /*O */
    ncf[ 3 * kd + 1 ] = 1; /*H */

    /*CO2 */
    ncf[ 4 * kd + 2 ] = 1; /*C */
    ncf[ 4 * kd + 0 ] = 2; /*O */

    /*HCO */
    ncf[ 5 * kd + 1 ] = 1; /*H */
    ncf[ 5 * kd + 2 ] = 1; /*C */
    ncf[ 5 * kd + 0 ] = 1; /*O */

}

void comp_qss_qfqr_coeff(double *  qf_co, double *  qr_co, double *  sc, double * qss_sc, double *  tc, double invT)
{

    /*reaction 1: O + HO2 => OH + O2 */
    qf_co[0] = sc[2]*qss_sc[2];
    qr_co[0] = 0.0;

    /*reaction 2: H + HO2 => O + H2O */
    qf_co[1] = sc[1]*qss_sc[2];
    qr_co[1] = 0.0;

    /*reaction 3: H + H2O2 <=> OH + H2O */
    qf_co[2] = sc[1]*qss_sc[3];
    qr_co[2] = sc[3]*qss_sc[1];

    /*reaction 4: O + CH => H + CO */
    qf_co[3] = sc[2]*qss_sc[5];
    qr_co[3] = 0.0;

    /*reaction 5: H + CH <=> C + H2 */
    qf_co[4] = sc[1]*qss_sc[5];
    qr_co[4] = qss_sc[4]*sc[0];

    /*reaction 6: O + CH2 <=> H + HCO */
    qf_co[5] = sc[2]*qss_sc[6];
    qr_co[5] = sc[1]*sc[5];

    /*reaction 7: H + O2 <=> O + OH */
    qf_co[6] = sc[1]*qss_sc[0];
    qr_co[6] = sc[2]*sc[3];

    /*reaction 8: H + HO2 <=> 2.000000 OH */
    qf_co[7] = sc[1]*qss_sc[2];
    qr_co[7] = pow(sc[3], 2.000000);

    /*reaction 9: OH + CO <=> H + CO2 */
    qf_co[8] = sc[3]*qss_sc[7];
    qr_co[8] = sc[1]*sc[4];

    /*reaction 10: OH + CH <=> H + HCO */
    qf_co[9] = sc[3]*qss_sc[5];
    qr_co[9] = sc[1]*sc[5];

    double T = tc[1];

    /*compute the mixture concentration */
    double mixture = 0.0;
    for (int i = 0; i < 6; ++i) {
        mixture += sc[i];
    }

    double Corr[10];
    for (int i = 0; i < 10; ++i) {
        Corr[i] = 1.0;
    }

    for (int i=0; i<10; i++)
    {
        qf_co[i] *= Corr[i] * k_f_save[i];
        qr_co[i] *= Corr[i] * k_f_save[i] / Kc_save[i];
    }

    return;
}
#endif

#ifndef AMREX_USE_CUDA
static double T_save = -1;
#ifdef _OPENMP
#pragma omp threadprivate(T_save)
#endif

static double k_f_save[10];
#ifdef _OPENMP
#pragma omp threadprivate(k_f_save)
#endif

static double Kc_save[10];
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

    double qdot, q_f[10], q_r[10];
    comp_qfqr(q_f, q_r, sc, tc, invT);

    for (int i = 0; i < 6; ++i) {
        wdot[i] = 0.0;
    }

    qdot = q_f[0]-q_r[0];
    wdot[2] -= qdot;
    wdot[3] += qdot;

    qdot = q_f[1]-q_r[1];
    wdot[1] -= qdot;
    wdot[2] += qdot;

    qdot = q_f[2]-q_r[2];
    wdot[1] -= qdot;
    wdot[3] += qdot;

    qdot = q_f[3]-q_r[3];
    wdot[1] += qdot;
    wdot[2] -= qdot;

    qdot = q_f[4]-q_r[4];
    wdot[0] += qdot;
    wdot[1] -= qdot;

    qdot = q_f[5]-q_r[5];
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[5] += qdot;

    qdot = q_f[6]-q_r[6];
    wdot[1] -= qdot;
    wdot[2] += qdot;
    wdot[3] += qdot;

    qdot = q_f[7]-q_r[7];
    wdot[1] -= qdot;
    wdot[3] += 2.000000 * qdot;

    qdot = q_f[8]-q_r[8];
    wdot[1] += qdot;
    wdot[3] -= qdot;
    wdot[4] += qdot;

    qdot = q_f[9]-q_r[9];
    wdot[1] += qdot;
    wdot[3] -= qdot;
    wdot[5] += qdot;

    return;
}

void comp_k_f(double *  tc, double invT, double *  k_f)
{
#ifdef __INTEL_COMPILER
    #pragma simd
#endif
    for (int i=0; i<10; ++i) {
        k_f[i] = prefactor_units[i] * fwd_A[i]
                    * exp(fwd_beta[i] * tc[0] - activation_units[i] * fwd_Ea[i] * invT);
    };
    return;
}

void comp_Kc(double *  tc, double invT, double *  Kc)
{
    /*compute the Gibbs free energy */
    double g_RT[6], g_RT_qss[8];
    gibbs(g_RT, tc);
    gibbs_qss(g_RT_qss, tc);

    Kc[0] = g_RT[2] - g_RT[3] - g_RT_qss[0] + g_RT_qss[2];
    Kc[1] = g_RT[1] - g_RT[2] - g_RT_qss[1] + g_RT_qss[2];
    Kc[2] = g_RT[1] - g_RT[3] - g_RT_qss[1] + g_RT_qss[3];
    Kc[3] = -g_RT[1] + g_RT[2] + g_RT_qss[5] - g_RT_qss[7];
    Kc[4] = -g_RT[0] + g_RT[1] - g_RT_qss[4] + g_RT_qss[5];
    Kc[5] = -g_RT[1] + g_RT[2] - g_RT[5] + g_RT_qss[6];
    Kc[6] = g_RT[1] - g_RT[2] - g_RT[3] + g_RT_qss[0];
    Kc[7] = g_RT[1] - 2.000000*g_RT[3] + g_RT_qss[2];
    Kc[8] = -g_RT[1] + g_RT[3] - g_RT[4] + g_RT_qss[7];
    Kc[9] = -g_RT[1] + g_RT[3] - g_RT[5] + g_RT_qss[5];

#ifdef __INTEL_COMPILER
     #pragma simd
#endif
    for (int i=0; i<10; ++i) {
        Kc[i] = exp(Kc[i]);
    };

    /*reference concentration: P_atm / (RT) in inverse mol/m^3 */
    double refC = 101325 / 8.31446 * invT;
    double refCinv = 1 / refC;


    return;
}

void comp_qfqr(double *  qf, double *  qr, double *  sc, double * qss_sc, double *  tc, double invT)
{

    /*reaction 1: O + HO2 => OH + O2 */
    qf[0] = sc[2]*qss_sc[2];
    qr[0] = 0.0;

    /*reaction 2: H + HO2 => O + H2O */
    qf[1] = sc[1]*qss_sc[2];
    qr[1] = 0.0;

    /*reaction 3: H + H2O2 <=> OH + H2O */
    qf[2] = sc[1]*qss_sc[3];
    qr[2] = sc[3]*qss_sc[1];

    /*reaction 4: O + CH => H + CO */
    qf[3] = sc[2]*qss_sc[5];
    qr[3] = 0.0;

    /*reaction 5: H + CH <=> C + H2 */
    qf[4] = sc[1]*qss_sc[5];
    qr[4] = qss_sc[4]*sc[0];

    /*reaction 6: O + CH2 <=> H + HCO */
    qf[5] = sc[2]*qss_sc[6];
    qr[5] = sc[1]*sc[5];

    /*reaction 7: H + O2 <=> O + OH */
    qf[6] = sc[1]*qss_sc[0];
    qr[6] = sc[2]*sc[3];

    /*reaction 8: H + HO2 <=> 2.000000 OH */
    qf[7] = sc[1]*qss_sc[2];
    qr[7] = pow(sc[3], 2.000000);

    /*reaction 9: OH + CO <=> H + CO2 */
    qf[8] = sc[3]*qss_sc[7];
    qr[8] = sc[1]*sc[4];

    /*reaction 10: OH + CH <=> H + HCO */
    qf[9] = sc[3]*qss_sc[5];
    qr[9] = sc[1]*sc[5];

    double T = tc[1];

    /*compute the mixture concentration */
    double mixture = 0.0;
    for (int i = 0; i < 6; ++i) {
        mixture += sc[i];
    }

    double Corr[10];
    for (int i = 0; i < 10; ++i) {
        Corr[i] = 1.0;
    }

    for (int i=0; i<10; i++)
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
    double c[6];

    for (int k=0; k<6; k++) {
        c[k] = 1.e6 * sc[k];
    }

    aJacobian_precond(J, c, *Tp, *HP);

    /* dwdot[k]/dT */
    /* dTdot/d[X] */
    for (int k=0; k<6; k++) {
        J[42+k] *= 1.e-6;
        J[k*7+6] *= 1.e6;
    }

    return;
}

/*compute the reaction Jacobian */
AMREX_GPU_HOST_DEVICE void DWDOT(double *  J, double *  sc, double *  Tp, int * consP)
{
    double c[6];

    for (int k=0; k<6; k++) {
        c[k] = 1.e6 * sc[k];
    }

    aJacobian(J, c, *Tp, *consP);

    /* dwdot[k]/dT */
    /* dTdot/d[X] */
    for (int k=0; k<6; k++) {
        J[42+k] *= 1.e-6;
        J[k*7+6] *= 1.e6;
    }

    return;
}

/*compute the sparsity pattern of the chemistry Jacobian */
AMREX_GPU_HOST_DEVICE void SPARSITY_INFO( int * nJdata, int * consP, int NCELLS)
{
    double c[6];
    double J[49];

    for (int k=0; k<6; k++) {
        c[k] = 1.0/ 6.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    int nJdata_tmp = 0;
    for (int k=0; k<7; k++) {
        for (int l=0; l<7; l++) {
            if(J[ 7 * k + l] != 0.0){
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
    double c[6];
    double J[49];

    for (int k=0; k<6; k++) {
        c[k] = 1.0/ 6.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    int nJdata_tmp = 0;
    for (int k=0; k<7; k++) {
        for (int l=0; l<7; l++) {
            if(k == l){
                nJdata_tmp = nJdata_tmp + 1;
            } else {
                if(J[ 7 * k + l] != 0.0){
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
    double c[6];
    double J[49];

    for (int k=0; k<6; k++) {
        c[k] = 1.0/ 6.000000 ;
    }

    aJacobian_precond(J, c, 1500.0, *consP);

    int nJdata_tmp = 0;
    for (int k=0; k<7; k++) {
        for (int l=0; l<7; l++) {
            if(k == l){
                nJdata_tmp = nJdata_tmp + 1;
            } else {
                if(J[ 7 * k + l] != 0.0){
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
    double c[6];
    double J[49];
    int offset_row;
    int offset_col;

    for (int k=0; k<6; k++) {
        c[k] = 1.0/ 6.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    colPtrs[0] = 0;
    int nJdata_tmp = 0;
    for (int nc=0; nc<NCELLS; nc++) {
        offset_row = nc * 7;
        offset_col = nc * 7;
        for (int k=0; k<7; k++) {
            for (int l=0; l<7; l++) {
                if(J[7*k + l] != 0.0) {
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
    double c[6];
    double J[49];
    int offset;

    for (int k=0; k<6; k++) {
        c[k] = 1.0/ 6.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    if (base == 1) {
        rowPtrs[0] = 1;
        int nJdata_tmp = 1;
        for (int nc=0; nc<NCELLS; nc++) {
            offset = nc * 7;
            for (int l=0; l<7; l++) {
                for (int k=0; k<7; k++) {
                    if(J[7*k + l] != 0.0) {
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
            offset = nc * 7;
            for (int l=0; l<7; l++) {
                for (int k=0; k<7; k++) {
                    if(J[7*k + l] != 0.0) {
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
    double c[6];
    double J[49];
    int offset;

    for (int k=0; k<6; k++) {
        c[k] = 1.0/ 6.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    if (base == 1) {
        rowPtr[0] = 1;
        int nJdata_tmp = 1;
        for (int nc=0; nc<NCELLS; nc++) {
            offset = nc * 7;
            for (int l=0; l<7; l++) {
                for (int k=0; k<7; k++) {
                    if (k == l) {
                        colVals[nJdata_tmp-1] = l+1 + offset; 
                        nJdata_tmp = nJdata_tmp + 1; 
                    } else {
                        if(J[7*k + l] != 0.0) {
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
            offset = nc * 7;
            for (int l=0; l<7; l++) {
                for (int k=0; k<7; k++) {
                    if (k == l) {
                        colVals[nJdata_tmp] = l + offset; 
                        nJdata_tmp = nJdata_tmp + 1; 
                    } else {
                        if(J[7*k + l] != 0.0) {
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
    double c[6];
    double J[49];

    for (int k=0; k<6; k++) {
        c[k] = 1.0/ 6.000000 ;
    }

    aJacobian_precond(J, c, 1500.0, *consP);

    colPtrs[0] = 0;
    int nJdata_tmp = 0;
    for (int k=0; k<7; k++) {
        for (int l=0; l<7; l++) {
            if (k == l) {
                rowVals[nJdata_tmp] = l; 
                indx[nJdata_tmp] = 7*k + l;
                nJdata_tmp = nJdata_tmp + 1; 
            } else {
                if(J[7*k + l] != 0.0) {
                    rowVals[nJdata_tmp] = l; 
                    indx[nJdata_tmp] = 7*k + l;
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
    double c[6];
    double J[49];

    for (int k=0; k<6; k++) {
        c[k] = 1.0/ 6.000000 ;
    }

    aJacobian_precond(J, c, 1500.0, *consP);

    if (base == 1) {
        rowPtr[0] = 1;
        int nJdata_tmp = 1;
        for (int l=0; l<7; l++) {
            for (int k=0; k<7; k++) {
                if (k == l) {
                    colVals[nJdata_tmp-1] = l+1; 
                    nJdata_tmp = nJdata_tmp + 1; 
                } else {
                    if(J[7*k + l] != 0.0) {
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
        for (int l=0; l<7; l++) {
            for (int k=0; k<7; k++) {
                if (k == l) {
                    colVals[nJdata_tmp] = l; 
                    nJdata_tmp = nJdata_tmp + 1; 
                } else {
                    if(J[7*k + l] != 0.0) {
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
    for (int i=0; i<49; i++) {
        J[i] = 0.0;
    }
}
#endif


/*compute an approx to the reaction Jacobian */
AMREX_GPU_HOST_DEVICE void aJacobian_precond(double *  J, double *  sc, double T, int HP)
{
    for (int i=0; i<49; i++) {
        J[i] = 0.0;
    }
}


/*compute d(Cp/R)/dT and d(Cv/R)/dT at the given temperature */
/*tc contains precomputed powers of T, tc[0] = log(T) */
AMREX_GPU_HOST_DEVICE void dcvpRdT(double * species, double *  tc)
{

    /*temperature */
    double T = tc[1];
    return;
}


/*compute the equilibrium constants for each reaction */
void equilibriumConstants(double *  kc, double *  g_RT, double T)
{
    /*reference concentration: P_atm / (RT) in inverse mol/m^3 */
    double refC = 101325 / 8.31446 / T;

    /*reaction 1: O + HO2 => OH + O2 */
    kc[0] = exp((g_RT[2] + g_RT_qss[2]) - (g_RT[3] + g_RT_qss[0]));

    /*reaction 2: H + HO2 => O + H2O */
    kc[1] = exp((g_RT[1] + g_RT_qss[2]) - (g_RT[2] + g_RT_qss[1]));

    /*reaction 3: H + H2O2 <=> OH + H2O */
    kc[2] = exp((g_RT[1] + g_RT_qss[3]) - (g_RT[3] + g_RT_qss[1]));

    /*reaction 4: O + CH => H + CO */
    kc[3] = exp((g_RT[2] + g_RT_qss[5]) - (g_RT[1] + g_RT_qss[7]));

    /*reaction 5: H + CH <=> C + H2 */
    kc[4] = exp((g_RT[1] + g_RT_qss[5]) - (g_RT_qss[4] + g_RT[0]));

    /*reaction 6: O + CH2 <=> H + HCO */
    kc[5] = exp((g_RT[2] + g_RT_qss[6]) - (g_RT[1] + g_RT[5]));

    /*reaction 7: H + O2 <=> O + OH */
    kc[6] = exp((g_RT[1] + g_RT_qss[0]) - (g_RT[2] + g_RT[3]));

    /*reaction 8: H + HO2 <=> 2.000000 OH */
    kc[7] = exp((g_RT[1] + g_RT_qss[2]) - (2.000000 * g_RT[3]));

    /*reaction 9: OH + CO <=> H + CO2 */
    kc[8] = exp((g_RT[3] + g_RT_qss[7]) - (g_RT[1] + g_RT[4]));

    /*reaction 10: OH + CH <=> H + HCO */
    kc[9] = exp((g_RT[3] + g_RT_qss[5]) - (g_RT[1] + g_RT[5]));

    return;
}


/*compute the g/(RT) at the given temperature */
/*tc contains precomputed powers of T, tc[0] = log(T) */
AMREX_GPU_HOST_DEVICE void gibbs(double * species, double *  tc)
{

    /*temperature */
    double T = tc[1];
    double invT = 1 / T;
    return;
}


/*compute the g/(RT) at the given temperature */
/*tc contains precomputed powers of T, tc[0] = log(T) */
AMREX_GPU_HOST_DEVICE void gibbs_qss(double * species, double *  tc)
{

    /*temperature */
    double T = tc[1];
    double invT = 1 / T;
    return;
}


/*compute the a/(RT) at the given temperature */
/*tc contains precomputed powers of T, tc[0] = log(T) */
AMREX_GPU_HOST_DEVICE void helmholtz(double * species, double *  tc)
{

    /*temperature */
    double T = tc[1];
    double invT = 1 / T;
    return;
}


/*compute Cv/R at the given temperature */
/*tc contains precomputed powers of T, tc[0] = log(T) */
AMREX_GPU_HOST_DEVICE void cv_R(double * species, double *  tc)
{

    /*temperature */
    double T = tc[1];
    return;
}


/*compute Cp/R at the given temperature */
/*tc contains precomputed powers of T, tc[0] = log(T) */
AMREX_GPU_HOST_DEVICE void cp_R(double * species, double *  tc)
{

    /*temperature */
    double T = tc[1];
    return;
}


/*compute the e/(RT) at the given temperature */
/*tc contains precomputed powers of T, tc[0] = log(T) */
AMREX_GPU_HOST_DEVICE void speciesInternalEnergy(double * species, double *  tc)
{

    /*temperature */
    double T = tc[1];
    double invT = 1 / T;
    return;
}


/*compute the h/(RT) at the given temperature (Eq 20) */
/*tc contains precomputed powers of T, tc[0] = log(T) */
AMREX_GPU_HOST_DEVICE void speciesEnthalpy(double * species, double *  tc)
{

    /*temperature */
    double T = tc[1];
    double invT = 1 / T;
    return;
}


/*compute the S/R at the given temperature (Eq 21) */
/*tc contains precomputed powers of T, tc[0] = log(T) */
AMREX_GPU_HOST_DEVICE void speciesEntropy(double * species, double *  tc)
{

    /*temperature */
    double T = tc[1];
    return;
}


/*save atomic weights into array */
void atomicWeight(double *  awt)
{
    awt[0] = 15.999400; /*O */
    awt[1] = 1.007970; /*H */
    awt[2] = 12.011150; /*C */

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
    *LENIMC = 26;}


void egtransetLENRMC(int* LENRMC ) {
    *LENRMC = 894;}


void egtransetNO(int* NO ) {
    *NO = 4;}


void egtransetKK(int* KK ) {
    *KK = 6;}


void egtransetNLITE(int* NLITE ) {
    *NLITE = 2;}


/*Patm in ergs/cm3 */
void egtransetPATM(double* PATM) {
    *PATM =   0.1013250000000000E+07;}


/*the molecular weights in g/mol */
void egtransetWT(double* WT ) {
    WT[0] = 2.01594000E+00;
    WT[1] = 1.00797000E+00;
    WT[2] = 1.59994000E+01;
    WT[3] = 1.70073700E+01;
    WT[4] = 4.40099500E+01;
    WT[5] = 2.90185200E+01;
}


/*the lennard-jones potential well depth eps/kb in K */
void egtransetEPS(double* EPS ) {
    EPS[0] = 3.80000000E+01;
    EPS[1] = 1.45000000E+02;
    EPS[2] = 8.00000000E+01;
    EPS[3] = 8.00000000E+01;
    EPS[4] = 2.44000000E+02;
    EPS[5] = 4.98000000E+02;
}


/*the lennard-jones collision diameter in Angstroms */
void egtransetSIG(double* SIG ) {
    SIG[0] = 2.92000000E+00;
    SIG[1] = 2.05000000E+00;
    SIG[2] = 2.75000000E+00;
    SIG[3] = 2.75000000E+00;
    SIG[4] = 3.76300000E+00;
    SIG[5] = 3.59000000E+00;
}


/*the dipole moment in Debye */
void egtransetDIP(double* DIP ) {
    DIP[0] = 0.00000000E+00;
    DIP[1] = 0.00000000E+00;
    DIP[2] = 0.00000000E+00;
    DIP[3] = 0.00000000E+00;
    DIP[4] = 0.00000000E+00;
    DIP[5] = 0.00000000E+00;
}


/*the polarizability in cubic Angstroms */
void egtransetPOL(double* POL ) {
    POL[0] = 7.90000000E-01;
    POL[1] = 0.00000000E+00;
    POL[2] = 0.00000000E+00;
    POL[3] = 0.00000000E+00;
    POL[4] = 2.65000000E+00;
    POL[5] = 0.00000000E+00;
}


/*the rotational relaxation collision number at 298 K */
void egtransetZROT(double* ZROT ) {
    ZROT[0] = 2.80000000E+02;
    ZROT[1] = 0.00000000E+00;
    ZROT[2] = 0.00000000E+00;
    ZROT[3] = 0.00000000E+00;
    ZROT[4] = 2.10000000E+00;
    ZROT[5] = 0.00000000E+00;
}


/*0: monoatomic, 1: linear, 2: nonlinear */
void egtransetNLIN(int* NLIN) {
    NLIN[0] = 1;
    NLIN[1] = 0;
    NLIN[2] = 0;
    NLIN[3] = 1;
    NLIN[4] = 1;
    NLIN[5] = 2;
}


/*Poly fits for the viscosities, dim NO*KK */
void egtransetCOFETA(double* COFETA) {
    COFETA[0] = -1.40419527E+01;
    COFETA[1] = 1.08789225E+00;
    COFETA[2] = -6.18592115E-02;
    COFETA[3] = 2.86838304E-03;
    COFETA[4] = -2.08449616E+01;
    COFETA[5] = 3.82794146E+00;
    COFETA[6] = -4.20999003E-01;
    COFETA[7] = 1.85596636E-02;
    COFETA[8] = -1.57832266E+01;
    COFETA[9] = 2.21311180E+00;
    COFETA[10] = -2.12959301E-01;
    COFETA[11] = 9.62195191E-03;
    COFETA[12] = -1.57526788E+01;
    COFETA[13] = 2.21311180E+00;
    COFETA[14] = -2.12959301E-01;
    COFETA[15] = 9.62195191E-03;
    COFETA[16] = -2.28110345E+01;
    COFETA[17] = 4.62954710E+00;
    COFETA[18] = -5.00689001E-01;
    COFETA[19] = 2.10012969E-02;
    COFETA[20] = -1.57492286E+01;
    COFETA[21] = 9.81716009E-01;
    COFETA[22] = 7.09938181E-02;
    COFETA[23] = -7.70023966E-03;
}


/*Poly fits for the conductivities, dim NO*KK */
void egtransetCOFLAM(double* COFLAM) {
    COFLAM[0] = 4.34729192E+00;
    COFLAM[1] = 1.55347646E+00;
    COFLAM[2] = -1.60615552E-01;
    COFLAM[3] = 9.89934485E-03;
    COFLAM[4] = -1.29505111E+00;
    COFLAM[5] = 3.82794146E+00;
    COFLAM[6] = -4.20999003E-01;
    COFLAM[7] = 1.85596636E-02;
    COFLAM[8] = 1.00207105E+00;
    COFLAM[9] = 2.21311180E+00;
    COFLAM[10] = -2.12959301E-01;
    COFLAM[11] = 9.62195191E-03;
    COFLAM[12] = 9.09561056E+00;
    COFLAM[13] = -1.12146910E+00;
    COFLAM[14] = 2.39077996E-01;
    COFLAM[15] = -9.74338015E-03;
    COFLAM[16] = -8.74831432E+00;
    COFLAM[17] = 4.79275291E+00;
    COFLAM[18] = -4.18685061E-01;
    COFLAM[19] = 1.35210242E-02;
    COFLAM[20] = 1.06317650E+01;
    COFLAM[21] = -3.96201529E+00;
    COFLAM[22] = 8.65922550E-01;
    COFLAM[23] = -4.80178076E-02;
}


/*Poly fits for the diffusion coefficients, dim NO*KK*KK */
void egtransetCOFD(double* COFD) {
    COFD[0] = -1.04285080E+01;
    COFD[1] = 2.23477534E+00;
    COFD[2] = -8.11809423E-02;
    COFD[3] = 3.77342041E-03;
    COFD[4] = -1.19370371E+01;
    COFD[5] = 2.99054752E+00;
    COFD[6] = -1.79624448E-01;
    COFD[7] = 8.03970815E-03;
    COFD[8] = -1.12653981E+01;
    COFD[9] = 2.43094296E+00;
    COFD[10] = -1.03798673E-01;
    COFD[11] = 4.60962717E-03;
    COFD[12] = -1.12687251E+01;
    COFD[13] = 2.43094296E+00;
    COFD[14] = -1.03798673E-01;
    COFD[15] = 4.60962717E-03;
    COFD[16] = -1.43978662E+01;
    COFD[17] = 3.49721576E+00;
    COFD[18] = -2.45465191E-01;
    COFD[19] = 1.08948372E-02;
    COFD[20] = -1.64242337E+01;
    COFD[21] = 4.26219537E+00;
    COFD[22] = -3.41524254E-01;
    COFD[23] = 1.49232070E-02;
    COFD[24] = -1.19370371E+01;
    COFD[25] = 2.99054752E+00;
    COFD[26] = -1.79624448E-01;
    COFD[27] = 8.03970815E-03;
    COFD[28] = -1.51395399E+01;
    COFD[29] = 4.36621619E+00;
    COFD[30] = -3.53866950E-01;
    COFD[31] = 1.54097445E-02;
    COFD[32] = -1.39677477E+01;
    COFD[33] = 3.71279442E+00;
    COFD[34] = -2.72718508E-01;
    COFD[35] = 1.20446550E-02;
    COFD[36] = -1.39695071E+01;
    COFD[37] = 3.71279442E+00;
    COFD[38] = -2.72718508E-01;
    COFD[39] = 1.20446550E-02;
    COFD[40] = -1.76943558E+01;
    COFD[41] = 4.88672714E+00;
    COFD[42] = -4.14778255E-01;
    COFD[43] = 1.77823474E-02;
    COFD[44] = -1.85331662E+01;
    COFD[45] = 5.04618677E+00;
    COFD[46] = -4.14801122E-01;
    COFD[47] = 1.69487185E-02;
    COFD[48] = -1.12653981E+01;
    COFD[49] = 2.43094296E+00;
    COFD[50] = -1.03798673E-01;
    COFD[51] = 4.60962717E-03;
    COFD[52] = -1.39677477E+01;
    COFD[53] = 3.71279442E+00;
    COFD[54] = -2.72718508E-01;
    COFD[55] = 1.20446550E-02;
    COFD[56] = -1.37174845E+01;
    COFD[57] = 3.11889373E+00;
    COFD[58] = -1.96402933E-01;
    COFD[59] = 8.77180880E-03;
    COFD[60] = -1.37325251E+01;
    COFD[61] = 3.11889373E+00;
    COFD[62] = -1.96402933E-01;
    COFD[63] = 8.77180880E-03;
    COFD[64] = -1.74119025E+01;
    COFD[65] = 4.28601449E+00;
    COFD[66] = -3.44182880E-01;
    COFD[67] = 1.50201783E-02;
    COFD[68] = -1.92502106E+01;
    COFD[69] = 4.94108508E+00;
    COFD[70] = -4.19008197E-01;
    COFD[71] = 1.78499457E-02;
    COFD[72] = -1.12687251E+01;
    COFD[73] = 2.43094296E+00;
    COFD[74] = -1.03798673E-01;
    COFD[75] = 4.60962717E-03;
    COFD[76] = -1.39695071E+01;
    COFD[77] = 3.71279442E+00;
    COFD[78] = -2.72718508E-01;
    COFD[79] = 1.20446550E-02;
    COFD[80] = -1.37325251E+01;
    COFD[81] = 3.11889373E+00;
    COFD[82] = -1.96402933E-01;
    COFD[83] = 8.77180880E-03;
    COFD[84] = -1.37480322E+01;
    COFD[85] = 3.11889373E+00;
    COFD[86] = -1.96402933E-01;
    COFD[87] = 8.77180880E-03;
    COFD[88] = -1.74341216E+01;
    COFD[89] = 4.28601449E+00;
    COFD[90] = -3.44182880E-01;
    COFD[91] = 1.50201783E-02;
    COFD[92] = -1.92696867E+01;
    COFD[93] = 4.94108508E+00;
    COFD[94] = -4.19008197E-01;
    COFD[95] = 1.78499457E-02;
    COFD[96] = -1.43978662E+01;
    COFD[97] = 3.49721576E+00;
    COFD[98] = -2.45465191E-01;
    COFD[99] = 1.08948372E-02;
    COFD[100] = -1.76943558E+01;
    COFD[101] = 4.88672714E+00;
    COFD[102] = -4.14778255E-01;
    COFD[103] = 1.77823474E-02;
    COFD[104] = -1.74119025E+01;
    COFD[105] = 4.28601449E+00;
    COFD[106] = -3.44182880E-01;
    COFD[107] = 1.50201783E-02;
    COFD[108] = -1.74341216E+01;
    COFD[109] = 4.28601449E+00;
    COFD[110] = -3.44182880E-01;
    COFD[111] = 1.50201783E-02;
    COFD[112] = -2.05810669E+01;
    COFD[113] = 5.07469434E+00;
    COFD[114] = -4.25340301E-01;
    COFD[115] = 1.76800795E-02;
    COFD[116] = -1.98561573E+01;
    COFD[117] = 4.57685026E+00;
    COFD[118] = -3.30016794E-01;
    COFD[119] = 1.23264865E-02;
    COFD[120] = -1.64242337E+01;
    COFD[121] = 4.26219537E+00;
    COFD[122] = -3.41524254E-01;
    COFD[123] = 1.49232070E-02;
    COFD[124] = -1.85331662E+01;
    COFD[125] = 5.04618677E+00;
    COFD[126] = -4.14801122E-01;
    COFD[127] = 1.69487185E-02;
    COFD[128] = -1.92502106E+01;
    COFD[129] = 4.94108508E+00;
    COFD[130] = -4.19008197E-01;
    COFD[131] = 1.78499457E-02;
    COFD[132] = -1.92696867E+01;
    COFD[133] = 4.94108508E+00;
    COFD[134] = -4.19008197E-01;
    COFD[135] = 1.78499457E-02;
    COFD[136] = -1.98561573E+01;
    COFD[137] = 4.57685026E+00;
    COFD[138] = -3.30016794E-01;
    COFD[139] = 1.23264865E-02;
    COFD[140] = -1.61938051E+01;
    COFD[141] = 2.80507926E+00;
    COFD[142] = -5.55394339E-02;
    COFD[143] = -1.30364179E-03;
}


/*List of specs with small weight, dim NLITE */
void egtransetKTDIF(int* KTDIF) {
    KTDIF[0] = 1;
    KTDIF[1] = 2;
}


/*Poly fits for thermal diff ratios, dim NO*NLITE*KK */
void egtransetCOFTD(double* COFTD) {
    COFTD[0] = 0.00000000E+00;
    COFTD[1] = 0.00000000E+00;
    COFTD[2] = 0.00000000E+00;
    COFTD[3] = 0.00000000E+00;
    COFTD[4] = -1.26715692E-01;
    COFTD[5] = -1.02530485E-04;
    COFTD[6] = 5.45604892E-08;
    COFTD[7] = -8.85181063E-12;
    COFTD[8] = 3.69825311E-01;
    COFTD[9] = 9.58927840E-05;
    COFTD[10] = -4.86954016E-08;
    COFTD[11] = 8.20765740E-12;
    COFTD[12] = 3.75475346E-01;
    COFTD[13] = 9.73577933E-05;
    COFTD[14] = -4.94393493E-08;
    COFTD[15] = 8.33305051E-12;
    COFTD[16] = 2.47129011E-01;
    COFTD[17] = 4.49395677E-04;
    COFTD[18] = -2.32030740E-07;
    COFTD[19] = 3.62578797E-11;
    COFTD[20] = 8.88011309E-02;
    COFTD[21] = 6.35611803E-04;
    COFTD[22] = -3.09074957E-07;
    COFTD[23] = 4.61052654E-11;
    COFTD[24] = 1.26715692E-01;
    COFTD[25] = 1.02530485E-04;
    COFTD[26] = -5.45604892E-08;
    COFTD[27] = 8.85181063E-12;
    COFTD[28] = 0.00000000E+00;
    COFTD[29] = 0.00000000E+00;
    COFTD[30] = 0.00000000E+00;
    COFTD[31] = 0.00000000E+00;
    COFTD[32] = 1.93107545E-01;
    COFTD[33] = 5.04123759E-04;
    COFTD[34] = -2.55796068E-07;
    COFTD[35] = 3.93758307E-11;
    COFTD[36] = 1.94560455E-01;
    COFTD[37] = 5.07916706E-04;
    COFTD[38] = -2.57720638E-07;
    COFTD[39] = 3.96720883E-11;
    COFTD[40] = -3.84966170E-02;
    COFTD[41] = 8.34794926E-04;
    COFTD[42] = -3.81031231E-07;
    COFTD[43] = 5.45531902E-11;
    COFTD[44] = -1.52598674E-01;
    COFTD[45] = 8.48164645E-04;
    COFTD[46] = -3.52118423E-07;
    COFTD[47] = 4.75234393E-11;
}

/* Replace this routine with the one generated by the Gauss Jordan solver of DW */
AMREX_GPU_HOST_DEVICE void sgjsolve(double* A, double* x, double* b) {
    amrex::Abort("sgjsolve not implemented, choose a different solver ");
}

/* Replace this routine with the one generated by the Gauss Jordan solver of DW */
AMREX_GPU_HOST_DEVICE void sgjsolve_simplified(double* A, double* x, double* b) {
    amrex::Abort("sgjsolve_simplified not implemented, choose a different solver ");
}

