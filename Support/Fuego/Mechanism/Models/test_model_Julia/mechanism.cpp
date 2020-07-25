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
    std::vector<std::vector<double>> kiv_qss(10); 
    std::vector<std::vector<double>> nuv_qss(10); 
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

/* Molecular weights */
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


/* Returns the char strings of element names */
void CKSYME(int * kname, int * plenkname )
{
    int i; /*Loop Counter */
    int lenkname = *plenkname;
    /*clear kname */
    for (i=0; i<lenkname*3; i++) {
        kname[i] = ' ';
    }

    /* O  */
    kname[ 0*lenkname + 0 ] = 'O';
    kname[ 0*lenkname + 1 ] = ' ';

    /* H  */
    kname[ 1*lenkname + 0 ] = 'H';
    kname[ 1*lenkname + 1 ] = ' ';

    /* C  */
    kname[ 2*lenkname + 0 ] = 'C';
    kname[ 2*lenkname + 1 ] = ' ';

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


/* Returns the char strings of species names */
void CKSYMS(int * kname, int * plenkname )
{
    int i; /*Loop Counter */
    int lenkname = *plenkname;
    /*clear kname */
    for (i=0; i<lenkname*6; i++) {
        kname[i] = ' ';
    }

    /* H2  */
    kname[ 0*lenkname + 0 ] = 'H';
    kname[ 0*lenkname + 1 ] = '2';
    kname[ 0*lenkname + 2 ] = ' ';

    /* H  */
    kname[ 1*lenkname + 0 ] = 'H';
    kname[ 1*lenkname + 1 ] = ' ';

    /* O  */
    kname[ 2*lenkname + 0 ] = 'O';
    kname[ 2*lenkname + 1 ] = ' ';

    /* OH  */
    kname[ 3*lenkname + 0 ] = 'O';
    kname[ 3*lenkname + 1 ] = 'H';
    kname[ 3*lenkname + 2 ] = ' ';

    /* CO2  */
    kname[ 4*lenkname + 0 ] = 'C';
    kname[ 4*lenkname + 1 ] = 'O';
    kname[ 4*lenkname + 2 ] = '2';
    kname[ 4*lenkname + 3 ] = ' ';

    /* HCO  */
    kname[ 5*lenkname + 0 ] = 'H';
    kname[ 5*lenkname + 1 ] = 'C';
    kname[ 5*lenkname + 2 ] = 'O';
    kname[ 5*lenkname + 3 ] = ' ';

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
    XW += x[0]*molecular_weights[0]; /*H2 */
    XW += x[1]*molecular_weights[1]; /*H */
    XW += x[2]*molecular_weights[2]; /*O */
    XW += x[3]*molecular_weights[3]; /*OH */
    XW += x[4]*molecular_weights[4]; /*CO2 */
    XW += x[5]*molecular_weights[5]; /*HCO */
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
    XW += x[0]*molecular_weights[0]; /*H2 */
    XW += x[1]*molecular_weights[1]; /*H */
    XW += x[2]*molecular_weights[2]; /*O */
    XW += x[3]*molecular_weights[3]; /*OH */
    XW += x[4]*molecular_weights[4]; /*CO2 */
    XW += x[5]*molecular_weights[5]; /*HCO */
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
    W += c[0]*molecular_weights[0]; /*H2 */
    W += c[1]*molecular_weights[1]; /*H */
    W += c[2]*molecular_weights[2]; /*O */
    W += c[3]*molecular_weights[3]; /*OH */
    W += c[4]*molecular_weights[4]; /*CO2 */
    W += c[5]*molecular_weights[5]; /*HCO */

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
/*of the species (mdim is num of elements) */
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
    double sc_qss[8];
    /* Fill sc_qss here*/
    comp_qss_sc(sc, sc_qss, tc, invT);
    comp_qfqr(q_f, q_r, sc, sc_qss, tc, invT);

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
    qr[4] = sc[0]*qss_sc[4];

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

void comp_qss_coeff(double *  qf_co, double *  qr_co, double *  sc, double *  tc, double invT)
{

    /*reaction 1: O + HO2 => OH + O2 */
    qf_co[0] = sc[2];
    qr_co[0] = 0.0;

    /*reaction 2: H + HO2 => O + H2O */
    qf_co[1] = sc[1];
    qr_co[1] = 0.0;

    /*reaction 3: H + H2O2 <=> OH + H2O */
    qf_co[2] = sc[1];
    qr_co[2] = sc[3];

    /*reaction 4: O + CH => H + CO */
    qf_co[3] = sc[2];
    qr_co[3] = 0.0;

    /*reaction 5: H + CH <=> C + H2 */
    qf_co[4] = sc[1];
    qr_co[4] = sc[0];

    /*reaction 6: O + CH2 <=> H + HCO */
    qf_co[5] = sc[2];
    qr_co[5] = sc[1]*sc[5];

    /*reaction 7: H + O2 <=> O + OH */
    qf_co[6] = sc[1];
    qr_co[6] = sc[2]*sc[3];

    /*reaction 8: H + HO2 <=> 2.000000 OH */
    qf_co[7] = sc[1];
    qr_co[7] = pow(sc[3], 2.000000);

    /*reaction 9: OH + CO <=> H + CO2 */
    qf_co[8] = sc[3];
    qr_co[8] = sc[1]*sc[4];

    /*reaction 10: OH + CH <=> H + HCO */
    qf_co[9] = sc[3];
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

void comp_qss_sc(double * sc, double * sc_qss, double * tc, double * invT)
{

    double  qf_co[10], qr_co[10];
    double epsilon = 1e-16

    comp_qss_coeff(qf_co, qr_co, sc, tc, invT);

    /*QSS species 2: HO2 */

    double HO2_num = epsilon - qr_co[7];
    double HO2_denom = epsilon - qf_co[0] - qf_co[1] - qf_co[7];

    sc_qss[2] = HO2_num/HO2_denom;



    /*QSS species 6: CH2 */

    double CH2_num = epsilon - qr_co[5];
    double CH2_denom = epsilon - qf_co[5];

    sc_qss[6] = CH2_num/CH2_denom;



    /*QSS species 4: C */

    double C_num = epsilon ;
    double C_denom = epsilon - qr_co[4];
    double C_rhs = C_num/C_denom;

    double C_CH = (epsilon + qf_co[4])/C_denom;

    /*QSS species 5: CH */

    double CH_num = epsilon - qr_co[9];
    double CH_denom = epsilon - qf_co[3] - qf_co[4] - qf_co[9];
    double CH_rhs = CH_num/CH_denom;

    double CH_C = (epsilon + qr_co[4])/CH_denom;

    sc_qss[4] = C_rhs - (C_CH * (CH_rhs - C_rhs * CH_C) / (1 - C_CH * CH_C));
    sc_qss[5] = (CH_rhs - C_rhs * CH_C) / (1 - C_CH * CH_C);



    /*QSS species 0: O2 */

    double O2_num = epsilon - qf_co[0]*sc_qss[0] - qr_co[6];
    double O2_denom = epsilon - qf_co[6];

    sc_qss[0] = O2_num/O2_denom;



    /*QSS species 7: CO */

    double CO_num = epsilon - qf_co[3]*sc_qss[7] - qr_co[8];
    double CO_denom = epsilon - qf_co[8];

    sc_qss[7] = CO_num/CO_denom;



    /*QSS species 1: H2O */

    double H2O_num = epsilon - qf_co[1]*sc_qss[1];
    double H2O_denom = epsilon - qr_co[2];
    double H2O_rhs = H2O_num/H2O_denom;

    double H2O_H2O2 = (epsilon + qf_co[2])/H2O_denom;

    /*QSS species 3: H2O2 */

    double H2O2_num = epsilon ;
    double H2O2_denom = epsilon - qf_co[2];
    double H2O2_rhs = H2O2_num/H2O2_denom;

    double H2O2_H2O = (epsilon + qr_co[2])/H2O2_denom;

    sc_qss[1] = H2O_rhs - (H2O_H2O2 * (H2O2_rhs - H2O_rhs * H2O2_H2O) / (1 - H2O_H2O2 * H2O2_H2O));
    sc_qss[3] = (H2O2_rhs - H2O_rhs * H2O2_H2O) / (1 - H2O_H2O2 * H2O2_H2O);



    return;
}

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

    /*species with midpoint at T=1000 kelvin */
    if (T < 1000) {
        /*species 0: H2 */
        species[0] =
            +7.98052075e-03
            -3.89563020e-05 * tc[1]
            +6.04716282e-08 * tc[2]
            -2.95044704e-11 * tc[3];
        /*species 1: H */
        species[1] =
            +7.05332819e-13
            -3.99183928e-15 * tc[1]
            +6.90244896e-18 * tc[2]
            -3.71092933e-21 * tc[3];
        /*species 2: O */
        species[2] =
            -3.27931884e-03
            +1.32861279e-05 * tc[1]
            -1.83841987e-08 * tc[2]
            +8.45063884e-12 * tc[3];
        /*species 3: OH */
        species[3] =
            -2.40131752e-03
            +9.23587682e-06 * tc[1]
            -1.16434000e-08 * tc[2]
            +5.45645880e-12 * tc[3];
        /*species 4: CO2 */
        species[4] =
            +8.98459677e-03
            -1.42471254e-05 * tc[1]
            +7.37757066e-09 * tc[2]
            -5.74798192e-13 * tc[3];
        /*species 5: HCO */
        species[5] =
            -3.24392532e-03
            +2.75598892e-05 * tc[1]
            -3.99432279e-08 * tc[2]
            +1.73507546e-11 * tc[3];
    } else {
        /*species 0: H2 */
        species[0] =
            -4.94024731e-05
            +9.98913556e-07 * tc[1]
            -5.38699182e-10 * tc[2]
            +8.01021504e-14 * tc[3];
        /*species 1: H */
        species[1] =
            -2.30842973e-11
            +3.23123896e-14 * tc[1]
            -1.42054571e-17 * tc[2]
            +1.99278943e-21 * tc[3];
        /*species 2: O */
        species[2] =
            -8.59741137e-05
            +8.38969178e-08 * tc[1]
            -3.00533397e-11 * tc[2]
            +4.91334764e-15 * tc[3];
        /*species 3: OH */
        species[3] =
            +5.48429716e-04
            +2.53010456e-07 * tc[1]
            -2.63838467e-10 * tc[2]
            +4.69649504e-14 * tc[3];
        /*species 4: CO2 */
        species[4] =
            +4.41437026e-03
            -4.42962808e-06 * tc[1]
            +1.57047056e-09 * tc[2]
            -1.88833666e-13 * tc[3];
        /*species 5: HCO */
        species[5] =
            +4.95695526e-03
            -4.96891226e-06 * tc[1]
            +1.76748533e-09 * tc[2]
            -2.13403484e-13 * tc[3];
    }
    return;
}


/*compute the equilibrium constants for each reaction */
void equilibriumConstants(double *  kc, double *  g_RT, double T)
{
    /*reference concentration: P_atm / (RT) in inverse mol/m^3 */
    double refC = 101325 / 8.31446 / T;

    /*reaction 1: O + HO2 => OH + O2 */
    kc[0] = exp((g_RT[2] + g_RT_qss[2]) - (g_RT[4] + g_RT_qss[0]));

    /*reaction 2: H + HO2 => O + H2O */
    kc[1] = exp((g_RT[1] + g_RT_qss[2]) - (g_RT[2] + g_RT_qss[1]));

    /*reaction 3: H + H2O2 <=> OH + H2O */
    kc[2] = exp((g_RT[1] + g_RT_qss[3]) - (g_RT[4] + g_RT_qss[1]));

    /*reaction 4: O + CH => H + CO */
    kc[3] = exp((g_RT[2] + g_RT_qss[5]) - (g_RT[1] + g_RT_qss[7]));

    /*reaction 5: H + CH <=> C + H2 */
    kc[4] = exp((g_RT[1] + g_RT_qss[5]) - (g_RT_qss[4] + g_RT[0]));

    /*reaction 6: O + CH2 <=> H + HCO */
    kc[5] = exp((g_RT[2] + g_RT_qss[6]) - (g_RT[1] + g_RT[13]));

    /*reaction 7: H + O2 <=> O + OH */
    kc[6] = exp((g_RT[1] + g_RT_qss[0]) - (g_RT[2] + g_RT[4]));

    /*reaction 8: H + HO2 <=> 2.000000 OH */
    kc[7] = exp((g_RT[1] + g_RT_qss[2]) - (2.000000 * g_RT[4]));

    /*reaction 9: OH + CO <=> H + CO2 */
    kc[8] = exp((g_RT[4] + g_RT_qss[7]) - (g_RT[1] + g_RT[12]));

    /*reaction 10: OH + CH <=> H + HCO */
    kc[9] = exp((g_RT[4] + g_RT_qss[5]) - (g_RT[1] + g_RT[13]));

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
        /*species 0: H2 */
        species[0] =
            -9.179351730000000e+02 * invT
            +1.661320882000000e+00
            -2.344331120000000e+00 * tc[0]
            -3.990260375000000e-03 * tc[1]
            +3.246358500000000e-06 * tc[2]
            -1.679767450000000e-09 * tc[3]
            +3.688058805000000e-13 * tc[4];
        /*species 1: H */
        species[1] =
            +2.547365990000000e+04 * invT
            +2.946682853000000e+00
            -2.500000000000000e+00 * tc[0]
            -3.526664095000000e-13 * tc[1]
            +3.326532733333333e-16 * tc[2]
            -1.917346933333333e-19 * tc[3]
            +4.638661660000000e-23 * tc[4];
        /*species 2: O */
        species[2] =
            +2.912225920000000e+04 * invT
            +1.116333640000000e+00
            -3.168267100000000e+00 * tc[0]
            +1.639659420000000e-03 * tc[1]
            -1.107177326666667e-06 * tc[2]
            +5.106721866666666e-10 * tc[3]
            -1.056329855000000e-13 * tc[4];
        /*species 3: OH */
        species[3] =
            +3.615080560000000e+03 * invT
            +4.095940888000000e+00
            -3.992015430000000e+00 * tc[0]
            +1.200658760000000e-03 * tc[1]
            -7.696564016666666e-07 * tc[2]
            +3.234277775000000e-10 * tc[3]
            -6.820573500000000e-14 * tc[4];
        /*species 4: CO2 */
        species[4] =
            -4.837196970000000e+04 * invT
            -7.544278700000000e+00
            -2.356773520000000e+00 * tc[0]
            -4.492298385000000e-03 * tc[1]
            +1.187260448333333e-06 * tc[2]
            -2.049325183333333e-10 * tc[3]
            +7.184977399999999e-15 * tc[4];
        /*species 5: HCO */
        species[5] =
            +3.839564960000000e+03 * invT
            +8.268134100000002e-01
            -4.221185840000000e+00 * tc[0]
            +1.621962660000000e-03 * tc[1]
            -2.296657433333333e-06 * tc[2]
            +1.109534108333333e-09 * tc[3]
            -2.168844325000000e-13 * tc[4];
    } else {
        /*species 0: H2 */
        species[0] =
            -9.501589220000000e+02 * invT
            +6.542302510000000e+00
            -3.337279200000000e+00 * tc[0]
            +2.470123655000000e-05 * tc[1]
            -8.324279633333333e-08 * tc[2]
            +1.496386616666667e-11 * tc[3]
            -1.001276880000000e-15 * tc[4];
        /*species 1: H */
        species[1] =
            +2.547365990000000e+04 * invT
            +2.946682924000000e+00
            -2.500000010000000e+00 * tc[0]
            +1.154214865000000e-11 * tc[1]
            -2.692699133333334e-15 * tc[2]
            +3.945960291666667e-19 * tc[3]
            -2.490986785000000e-23 * tc[4];
        /*species 2: O */
        species[2] =
            +2.921757910000000e+04 * invT
            -2.214917859999999e+00
            -2.569420780000000e+00 * tc[0]
            +4.298705685000000e-05 * tc[1]
            -6.991409816666667e-09 * tc[2]
            +8.348149916666666e-13 * tc[3]
            -6.141684549999999e-17 * tc[4];
        /*species 3: OH */
        species[3] =
            +3.858657000000000e+03 * invT
            -1.383808430000000e+00
            -3.092887670000000e+00 * tc[0]
            -2.742148580000000e-04 * tc[1]
            -2.108420466666667e-08 * tc[2]
            +7.328846300000000e-12 * tc[3]
            -5.870618800000000e-16 * tc[4];
        /*species 4: CO2 */
        species[4] =
            -4.875916600000000e+04 * invT
            +1.585822230000000e+00
            -3.857460290000000e+00 * tc[0]
            -2.207185130000000e-03 * tc[1]
            +3.691356733333334e-07 * tc[2]
            -4.362418233333334e-11 * tc[3]
            +2.360420820000000e-15 * tc[4];
        /*species 5: HCO */
        species[5] =
            +4.011918150000000e+03 * invT
            -7.026170540000000e+00
            -2.772174380000000e+00 * tc[0]
            -2.478477630000000e-03 * tc[1]
            +4.140760216666667e-07 * tc[2]
            -4.909681483333334e-11 * tc[3]
            +2.667543555000000e-15 * tc[4];
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
        /*species 0: O2 */
        species[0] =
            -1.063943560000000e+03 * invT
            +1.247806300000001e-01
            -3.782456360000000e+00 * tc[0]
            +1.498367080000000e-03 * tc[1]
            -1.641217001666667e-06 * tc[2]
            +8.067745908333334e-10 * tc[3]
            -1.621864185000000e-13 * tc[4];
        /*species 1: H2O */
        species[1] =
            -3.029372670000000e+04 * invT
            +5.047672768000000e+00
            -4.198640560000000e+00 * tc[0]
            +1.018217050000000e-03 * tc[1]
            -1.086733685000000e-06 * tc[2]
            +4.573308850000000e-10 * tc[3]
            -8.859890850000000e-14 * tc[4];
        /*species 2: HO2 */
        species[2] =
            +2.948080400000000e+02 * invT
            +5.851355599999999e-01
            -4.301798010000000e+00 * tc[0]
            +2.374560255000000e-03 * tc[1]
            -3.526381516666666e-06 * tc[2]
            +2.023032450000000e-09 * tc[3]
            -4.646125620000001e-13 * tc[4];
        /*species 3: H2O2 */
        species[3] =
            -1.770258210000000e+04 * invT
            +8.410619499999998e-01
            -4.276112690000000e+00 * tc[0]
            +2.714112085000000e-04 * tc[1]
            -2.788928350000000e-06 * tc[2]
            +1.798090108333333e-09 * tc[3]
            -4.312271815000000e-13 * tc[4];
        /*species 4: C */
        species[4] =
            +8.544388320000000e+04 * invT
            -1.977068930000000e+00
            -2.554239550000000e+00 * tc[0]
            +1.607688620000000e-04 * tc[1]
            -1.222987075000000e-07 * tc[2]
            +6.101957408333333e-11 * tc[3]
            -1.332607230000000e-14 * tc[4];
        /*species 5: CH */
        species[5] =
            +7.079729340000000e+04 * invT
            +1.405805570000000e+00
            -3.489816650000000e+00 * tc[0]
            -1.619177705000000e-04 * tc[1]
            +2.814984416666667e-07 * tc[2]
            -2.635144391666666e-10 * tc[3]
            +7.030453350000001e-14 * tc[4];
        /*species 6: CH2 */
        species[6] =
            +4.600404010000000e+04 * invT
            +2.200146820000000e+00
            -3.762678670000000e+00 * tc[0]
            -4.844360715000000e-04 * tc[1]
            -4.658164016666667e-07 * tc[2]
            +3.209092941666667e-10 * tc[3]
            -8.437085950000000e-14 * tc[4];
        /*species 7: CO */
        species[7] =
            -1.434408600000000e+04 * invT
            +7.112418999999992e-02
            -3.579533470000000e+00 * tc[0]
            +3.051768400000000e-04 * tc[1]
            -1.694690550000000e-07 * tc[2]
            -7.558382366666667e-11 * tc[3]
            +4.522122495000000e-14 * tc[4];
    } else {
        /*species 0: O2 */
        species[0] =
            -1.088457720000000e+03 * invT
            -2.170693450000000e+00
            -3.282537840000000e+00 * tc[0]
            -7.415437700000000e-04 * tc[1]
            +1.263277781666667e-07 * tc[2]
            -1.745587958333333e-11 * tc[3]
            +1.083588970000000e-15 * tc[4];
        /*species 1: H2O */
        species[1] =
            -3.000429710000000e+04 * invT
            -1.932777610000000e+00
            -3.033992490000000e+00 * tc[0]
            -1.088459020000000e-03 * tc[1]
            +2.734541966666666e-08 * tc[2]
            +8.086832250000000e-12 * tc[3]
            -8.410049600000000e-16 * tc[4];
        /*species 2: HO2 */
        species[2] =
            +1.118567130000000e+02 * invT
            +2.321087500000001e-01
            -4.017210900000000e+00 * tc[0]
            -1.119910065000000e-03 * tc[1]
            +1.056096916666667e-07 * tc[2]
            -9.520530833333334e-12 * tc[3]
            +5.395426750000000e-16 * tc[4];
        /*species 3: H2O2 */
        species[3] =
            -1.786178770000000e+04 * invT
            +1.248846229999999e+00
            -4.165002850000000e+00 * tc[0]
            -2.454158470000000e-03 * tc[1]
            +3.168987083333333e-07 * tc[2]
            -3.093216550000000e-11 * tc[3]
            +1.439541525000000e-15 * tc[4];
        /*species 4: C */
        species[4] =
            +8.545129530000000e+04 * invT
            -2.308834850000000e+00
            -2.492668880000000e+00 * tc[0]
            -2.399446420000000e-05 * tc[1]
            +1.207225033333333e-08 * tc[2]
            -3.119091908333333e-12 * tc[3]
            +2.436389465000000e-16 * tc[4];
        /*species 5: CH */
        species[5] =
            +7.101243640000001e+04 * invT
            -2.606515260000000e+00
            -2.878464730000000e+00 * tc[0]
            -4.854568405000000e-04 * tc[1]
            -2.407427583333333e-08 * tc[2]
            +1.089065408333333e-11 * tc[3]
            -8.803969149999999e-16 * tc[4];
        /*species 6: CH2 */
        species[6] =
            +4.626360400000000e+04 * invT
            -3.297092110000000e+00
            -2.874101130000000e+00 * tc[0]
            -1.828196460000000e-03 * tc[1]
            +2.348243283333333e-07 * tc[2]
            -2.168162908333333e-11 * tc[3]
            +9.386378350000000e-16 * tc[4];
        /*species 7: CO */
        species[7] =
            -1.415187240000000e+04 * invT
            -5.103502110000000e+00
            -2.715185610000000e+00 * tc[0]
            -1.031263715000000e-03 * tc[1]
            +1.664709618333334e-07 * tc[2]
            -1.917108400000000e-11 * tc[3]
            +1.018238580000000e-15 * tc[4];
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
        /*species 0: H2 */
        species[0] =
            -9.17935173e+02 * invT
            +6.61320882e-01
            -2.34433112e+00 * tc[0]
            -3.99026037e-03 * tc[1]
            +3.24635850e-06 * tc[2]
            -1.67976745e-09 * tc[3]
            +3.68805881e-13 * tc[4];
        /*species 1: H */
        species[1] =
            +2.54736599e+04 * invT
            +1.94668285e+00
            -2.50000000e+00 * tc[0]
            -3.52666409e-13 * tc[1]
            +3.32653273e-16 * tc[2]
            -1.91734693e-19 * tc[3]
            +4.63866166e-23 * tc[4];
        /*species 2: O */
        species[2] =
            +2.91222592e+04 * invT
            +1.16333640e-01
            -3.16826710e+00 * tc[0]
            +1.63965942e-03 * tc[1]
            -1.10717733e-06 * tc[2]
            +5.10672187e-10 * tc[3]
            -1.05632985e-13 * tc[4];
        /*species 3: OH */
        species[3] =
            +3.61508056e+03 * invT
            +3.09594089e+00
            -3.99201543e+00 * tc[0]
            +1.20065876e-03 * tc[1]
            -7.69656402e-07 * tc[2]
            +3.23427778e-10 * tc[3]
            -6.82057350e-14 * tc[4];
        /*species 4: CO2 */
        species[4] =
            -4.83719697e+04 * invT
            -8.54427870e+00
            -2.35677352e+00 * tc[0]
            -4.49229839e-03 * tc[1]
            +1.18726045e-06 * tc[2]
            -2.04932518e-10 * tc[3]
            +7.18497740e-15 * tc[4];
        /*species 5: HCO */
        species[5] =
            +3.83956496e+03 * invT
            -1.73186590e-01
            -4.22118584e+00 * tc[0]
            +1.62196266e-03 * tc[1]
            -2.29665743e-06 * tc[2]
            +1.10953411e-09 * tc[3]
            -2.16884432e-13 * tc[4];
    } else {
        /*species 0: H2 */
        species[0] =
            -9.50158922e+02 * invT
            +5.54230251e+00
            -3.33727920e+00 * tc[0]
            +2.47012365e-05 * tc[1]
            -8.32427963e-08 * tc[2]
            +1.49638662e-11 * tc[3]
            -1.00127688e-15 * tc[4];
        /*species 1: H */
        species[1] =
            +2.54736599e+04 * invT
            +1.94668292e+00
            -2.50000001e+00 * tc[0]
            +1.15421486e-11 * tc[1]
            -2.69269913e-15 * tc[2]
            +3.94596029e-19 * tc[3]
            -2.49098679e-23 * tc[4];
        /*species 2: O */
        species[2] =
            +2.92175791e+04 * invT
            -3.21491786e+00
            -2.56942078e+00 * tc[0]
            +4.29870569e-05 * tc[1]
            -6.99140982e-09 * tc[2]
            +8.34814992e-13 * tc[3]
            -6.14168455e-17 * tc[4];
        /*species 3: OH */
        species[3] =
            +3.85865700e+03 * invT
            -2.38380843e+00
            -3.09288767e+00 * tc[0]
            -2.74214858e-04 * tc[1]
            -2.10842047e-08 * tc[2]
            +7.32884630e-12 * tc[3]
            -5.87061880e-16 * tc[4];
        /*species 4: CO2 */
        species[4] =
            -4.87591660e+04 * invT
            +5.85822230e-01
            -3.85746029e+00 * tc[0]
            -2.20718513e-03 * tc[1]
            +3.69135673e-07 * tc[2]
            -4.36241823e-11 * tc[3]
            +2.36042082e-15 * tc[4];
        /*species 5: HCO */
        species[5] =
            +4.01191815e+03 * invT
            -8.02617054e+00
            -2.77217438e+00 * tc[0]
            -2.47847763e-03 * tc[1]
            +4.14076022e-07 * tc[2]
            -4.90968148e-11 * tc[3]
            +2.66754356e-15 * tc[4];
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
        /*species 0: H2 */
        species[0] =
            +1.34433112e+00
            +7.98052075e-03 * tc[1]
            -1.94781510e-05 * tc[2]
            +2.01572094e-08 * tc[3]
            -7.37611761e-12 * tc[4];
        /*species 1: H */
        species[1] =
            +1.50000000e+00
            +7.05332819e-13 * tc[1]
            -1.99591964e-15 * tc[2]
            +2.30081632e-18 * tc[3]
            -9.27732332e-22 * tc[4];
        /*species 2: O */
        species[2] =
            +2.16826710e+00
            -3.27931884e-03 * tc[1]
            +6.64306396e-06 * tc[2]
            -6.12806624e-09 * tc[3]
            +2.11265971e-12 * tc[4];
        /*species 3: OH */
        species[3] =
            +2.99201543e+00
            -2.40131752e-03 * tc[1]
            +4.61793841e-06 * tc[2]
            -3.88113333e-09 * tc[3]
            +1.36411470e-12 * tc[4];
        /*species 4: CO2 */
        species[4] =
            +1.35677352e+00
            +8.98459677e-03 * tc[1]
            -7.12356269e-06 * tc[2]
            +2.45919022e-09 * tc[3]
            -1.43699548e-13 * tc[4];
        /*species 5: HCO */
        species[5] =
            +3.22118584e+00
            -3.24392532e-03 * tc[1]
            +1.37799446e-05 * tc[2]
            -1.33144093e-08 * tc[3]
            +4.33768865e-12 * tc[4];
    } else {
        /*species 0: H2 */
        species[0] =
            +2.33727920e+00
            -4.94024731e-05 * tc[1]
            +4.99456778e-07 * tc[2]
            -1.79566394e-10 * tc[3]
            +2.00255376e-14 * tc[4];
        /*species 1: H */
        species[1] =
            +1.50000001e+00
            -2.30842973e-11 * tc[1]
            +1.61561948e-14 * tc[2]
            -4.73515235e-18 * tc[3]
            +4.98197357e-22 * tc[4];
        /*species 2: O */
        species[2] =
            +1.56942078e+00
            -8.59741137e-05 * tc[1]
            +4.19484589e-08 * tc[2]
            -1.00177799e-11 * tc[3]
            +1.22833691e-15 * tc[4];
        /*species 3: OH */
        species[3] =
            +2.09288767e+00
            +5.48429716e-04 * tc[1]
            +1.26505228e-07 * tc[2]
            -8.79461556e-11 * tc[3]
            +1.17412376e-14 * tc[4];
        /*species 4: CO2 */
        species[4] =
            +2.85746029e+00
            +4.41437026e-03 * tc[1]
            -2.21481404e-06 * tc[2]
            +5.23490188e-10 * tc[3]
            -4.72084164e-14 * tc[4];
        /*species 5: HCO */
        species[5] =
            +1.77217438e+00
            +4.95695526e-03 * tc[1]
            -2.48445613e-06 * tc[2]
            +5.89161778e-10 * tc[3]
            -5.33508711e-14 * tc[4];
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
        /*species 0: H2 */
        species[0] =
            +2.34433112e+00
            +7.98052075e-03 * tc[1]
            -1.94781510e-05 * tc[2]
            +2.01572094e-08 * tc[3]
            -7.37611761e-12 * tc[4];
        /*species 1: H */
        species[1] =
            +2.50000000e+00
            +7.05332819e-13 * tc[1]
            -1.99591964e-15 * tc[2]
            +2.30081632e-18 * tc[3]
            -9.27732332e-22 * tc[4];
        /*species 2: O */
        species[2] =
            +3.16826710e+00
            -3.27931884e-03 * tc[1]
            +6.64306396e-06 * tc[2]
            -6.12806624e-09 * tc[3]
            +2.11265971e-12 * tc[4];
        /*species 3: OH */
        species[3] =
            +3.99201543e+00
            -2.40131752e-03 * tc[1]
            +4.61793841e-06 * tc[2]
            -3.88113333e-09 * tc[3]
            +1.36411470e-12 * tc[4];
        /*species 4: CO2 */
        species[4] =
            +2.35677352e+00
            +8.98459677e-03 * tc[1]
            -7.12356269e-06 * tc[2]
            +2.45919022e-09 * tc[3]
            -1.43699548e-13 * tc[4];
        /*species 5: HCO */
        species[5] =
            +4.22118584e+00
            -3.24392532e-03 * tc[1]
            +1.37799446e-05 * tc[2]
            -1.33144093e-08 * tc[3]
            +4.33768865e-12 * tc[4];
    } else {
        /*species 0: H2 */
        species[0] =
            +3.33727920e+00
            -4.94024731e-05 * tc[1]
            +4.99456778e-07 * tc[2]
            -1.79566394e-10 * tc[3]
            +2.00255376e-14 * tc[4];
        /*species 1: H */
        species[1] =
            +2.50000001e+00
            -2.30842973e-11 * tc[1]
            +1.61561948e-14 * tc[2]
            -4.73515235e-18 * tc[3]
            +4.98197357e-22 * tc[4];
        /*species 2: O */
        species[2] =
            +2.56942078e+00
            -8.59741137e-05 * tc[1]
            +4.19484589e-08 * tc[2]
            -1.00177799e-11 * tc[3]
            +1.22833691e-15 * tc[4];
        /*species 3: OH */
        species[3] =
            +3.09288767e+00
            +5.48429716e-04 * tc[1]
            +1.26505228e-07 * tc[2]
            -8.79461556e-11 * tc[3]
            +1.17412376e-14 * tc[4];
        /*species 4: CO2 */
        species[4] =
            +3.85746029e+00
            +4.41437026e-03 * tc[1]
            -2.21481404e-06 * tc[2]
            +5.23490188e-10 * tc[3]
            -4.72084164e-14 * tc[4];
        /*species 5: HCO */
        species[5] =
            +2.77217438e+00
            +4.95695526e-03 * tc[1]
            -2.48445613e-06 * tc[2]
            +5.89161778e-10 * tc[3]
            -5.33508711e-14 * tc[4];
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
        /*species 0: H2 */
        species[0] =
            +1.34433112e+00
            +3.99026037e-03 * tc[1]
            -6.49271700e-06 * tc[2]
            +5.03930235e-09 * tc[3]
            -1.47522352e-12 * tc[4]
            -9.17935173e+02 * invT;
        /*species 1: H */
        species[1] =
            +1.50000000e+00
            +3.52666409e-13 * tc[1]
            -6.65306547e-16 * tc[2]
            +5.75204080e-19 * tc[3]
            -1.85546466e-22 * tc[4]
            +2.54736599e+04 * invT;
        /*species 2: O */
        species[2] =
            +2.16826710e+00
            -1.63965942e-03 * tc[1]
            +2.21435465e-06 * tc[2]
            -1.53201656e-09 * tc[3]
            +4.22531942e-13 * tc[4]
            +2.91222592e+04 * invT;
        /*species 3: OH */
        species[3] =
            +2.99201543e+00
            -1.20065876e-03 * tc[1]
            +1.53931280e-06 * tc[2]
            -9.70283332e-10 * tc[3]
            +2.72822940e-13 * tc[4]
            +3.61508056e+03 * invT;
        /*species 4: CO2 */
        species[4] =
            +1.35677352e+00
            +4.49229839e-03 * tc[1]
            -2.37452090e-06 * tc[2]
            +6.14797555e-10 * tc[3]
            -2.87399096e-14 * tc[4]
            -4.83719697e+04 * invT;
        /*species 5: HCO */
        species[5] =
            +3.22118584e+00
            -1.62196266e-03 * tc[1]
            +4.59331487e-06 * tc[2]
            -3.32860233e-09 * tc[3]
            +8.67537730e-13 * tc[4]
            +3.83956496e+03 * invT;
    } else {
        /*species 0: H2 */
        species[0] =
            +2.33727920e+00
            -2.47012365e-05 * tc[1]
            +1.66485593e-07 * tc[2]
            -4.48915985e-11 * tc[3]
            +4.00510752e-15 * tc[4]
            -9.50158922e+02 * invT;
        /*species 1: H */
        species[1] =
            +1.50000001e+00
            -1.15421486e-11 * tc[1]
            +5.38539827e-15 * tc[2]
            -1.18378809e-18 * tc[3]
            +9.96394714e-23 * tc[4]
            +2.54736599e+04 * invT;
        /*species 2: O */
        species[2] =
            +1.56942078e+00
            -4.29870569e-05 * tc[1]
            +1.39828196e-08 * tc[2]
            -2.50444497e-12 * tc[3]
            +2.45667382e-16 * tc[4]
            +2.92175791e+04 * invT;
        /*species 3: OH */
        species[3] =
            +2.09288767e+00
            +2.74214858e-04 * tc[1]
            +4.21684093e-08 * tc[2]
            -2.19865389e-11 * tc[3]
            +2.34824752e-15 * tc[4]
            +3.85865700e+03 * invT;
        /*species 4: CO2 */
        species[4] =
            +2.85746029e+00
            +2.20718513e-03 * tc[1]
            -7.38271347e-07 * tc[2]
            +1.30872547e-10 * tc[3]
            -9.44168328e-15 * tc[4]
            -4.87591660e+04 * invT;
        /*species 5: HCO */
        species[5] =
            +1.77217438e+00
            +2.47847763e-03 * tc[1]
            -8.28152043e-07 * tc[2]
            +1.47290445e-10 * tc[3]
            -1.06701742e-14 * tc[4]
            +4.01191815e+03 * invT;
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
        /*species 0: H2 */
        species[0] =
            +2.34433112e+00
            +3.99026037e-03 * tc[1]
            -6.49271700e-06 * tc[2]
            +5.03930235e-09 * tc[3]
            -1.47522352e-12 * tc[4]
            -9.17935173e+02 * invT;
        /*species 1: H */
        species[1] =
            +2.50000000e+00
            +3.52666409e-13 * tc[1]
            -6.65306547e-16 * tc[2]
            +5.75204080e-19 * tc[3]
            -1.85546466e-22 * tc[4]
            +2.54736599e+04 * invT;
        /*species 2: O */
        species[2] =
            +3.16826710e+00
            -1.63965942e-03 * tc[1]
            +2.21435465e-06 * tc[2]
            -1.53201656e-09 * tc[3]
            +4.22531942e-13 * tc[4]
            +2.91222592e+04 * invT;
        /*species 3: OH */
        species[3] =
            +3.99201543e+00
            -1.20065876e-03 * tc[1]
            +1.53931280e-06 * tc[2]
            -9.70283332e-10 * tc[3]
            +2.72822940e-13 * tc[4]
            +3.61508056e+03 * invT;
        /*species 4: CO2 */
        species[4] =
            +2.35677352e+00
            +4.49229839e-03 * tc[1]
            -2.37452090e-06 * tc[2]
            +6.14797555e-10 * tc[3]
            -2.87399096e-14 * tc[4]
            -4.83719697e+04 * invT;
        /*species 5: HCO */
        species[5] =
            +4.22118584e+00
            -1.62196266e-03 * tc[1]
            +4.59331487e-06 * tc[2]
            -3.32860233e-09 * tc[3]
            +8.67537730e-13 * tc[4]
            +3.83956496e+03 * invT;
    } else {
        /*species 0: H2 */
        species[0] =
            +3.33727920e+00
            -2.47012365e-05 * tc[1]
            +1.66485593e-07 * tc[2]
            -4.48915985e-11 * tc[3]
            +4.00510752e-15 * tc[4]
            -9.50158922e+02 * invT;
        /*species 1: H */
        species[1] =
            +2.50000001e+00
            -1.15421486e-11 * tc[1]
            +5.38539827e-15 * tc[2]
            -1.18378809e-18 * tc[3]
            +9.96394714e-23 * tc[4]
            +2.54736599e+04 * invT;
        /*species 2: O */
        species[2] =
            +2.56942078e+00
            -4.29870569e-05 * tc[1]
            +1.39828196e-08 * tc[2]
            -2.50444497e-12 * tc[3]
            +2.45667382e-16 * tc[4]
            +2.92175791e+04 * invT;
        /*species 3: OH */
        species[3] =
            +3.09288767e+00
            +2.74214858e-04 * tc[1]
            +4.21684093e-08 * tc[2]
            -2.19865389e-11 * tc[3]
            +2.34824752e-15 * tc[4]
            +3.85865700e+03 * invT;
        /*species 4: CO2 */
        species[4] =
            +3.85746029e+00
            +2.20718513e-03 * tc[1]
            -7.38271347e-07 * tc[2]
            +1.30872547e-10 * tc[3]
            -9.44168328e-15 * tc[4]
            -4.87591660e+04 * invT;
        /*species 5: HCO */
        species[5] =
            +2.77217438e+00
            +2.47847763e-03 * tc[1]
            -8.28152043e-07 * tc[2]
            +1.47290445e-10 * tc[3]
            -1.06701742e-14 * tc[4]
            +4.01191815e+03 * invT;
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
        /*species 0: H2 */
        species[0] =
            +2.34433112e+00 * tc[0]
            +7.98052075e-03 * tc[1]
            -9.73907550e-06 * tc[2]
            +6.71906980e-09 * tc[3]
            -1.84402940e-12 * tc[4]
            +6.83010238e-01 ;
        /*species 1: H */
        species[1] =
            +2.50000000e+00 * tc[0]
            +7.05332819e-13 * tc[1]
            -9.97959820e-16 * tc[2]
            +7.66938773e-19 * tc[3]
            -2.31933083e-22 * tc[4]
            -4.46682853e-01 ;
        /*species 2: O */
        species[2] =
            +3.16826710e+00 * tc[0]
            -3.27931884e-03 * tc[1]
            +3.32153198e-06 * tc[2]
            -2.04268875e-09 * tc[3]
            +5.28164927e-13 * tc[4]
            +2.05193346e+00 ;
        /*species 3: OH */
        species[3] =
            +3.99201543e+00 * tc[0]
            -2.40131752e-03 * tc[1]
            +2.30896920e-06 * tc[2]
            -1.29371111e-09 * tc[3]
            +3.41028675e-13 * tc[4]
            -1.03925458e-01 ;
        /*species 4: CO2 */
        species[4] =
            +2.35677352e+00 * tc[0]
            +8.98459677e-03 * tc[1]
            -3.56178134e-06 * tc[2]
            +8.19730073e-10 * tc[3]
            -3.59248870e-14 * tc[4]
            +9.90105222e+00 ;
        /*species 5: HCO */
        species[5] =
            +4.22118584e+00 * tc[0]
            -3.24392532e-03 * tc[1]
            +6.88997230e-06 * tc[2]
            -4.43813643e-09 * tc[3]
            +1.08442216e-12 * tc[4]
            +3.39437243e+00 ;
    } else {
        /*species 0: H2 */
        species[0] =
            +3.33727920e+00 * tc[0]
            -4.94024731e-05 * tc[1]
            +2.49728389e-07 * tc[2]
            -5.98554647e-11 * tc[3]
            +5.00638440e-15 * tc[4]
            -3.20502331e+00 ;
        /*species 1: H */
        species[1] =
            +2.50000001e+00 * tc[0]
            -2.30842973e-11 * tc[1]
            +8.07809740e-15 * tc[2]
            -1.57838412e-18 * tc[3]
            +1.24549339e-22 * tc[4]
            -4.46682914e-01 ;
        /*species 2: O */
        species[2] =
            +2.56942078e+00 * tc[0]
            -8.59741137e-05 * tc[1]
            +2.09742295e-08 * tc[2]
            -3.33925997e-12 * tc[3]
            +3.07084227e-16 * tc[4]
            +4.78433864e+00 ;
        /*species 3: OH */
        species[3] =
            +3.09288767e+00 * tc[0]
            +5.48429716e-04 * tc[1]
            +6.32526140e-08 * tc[2]
            -2.93153852e-11 * tc[3]
            +2.93530940e-15 * tc[4]
            +4.47669610e+00 ;
        /*species 4: CO2 */
        species[4] =
            +3.85746029e+00 * tc[0]
            +4.41437026e-03 * tc[1]
            -1.10740702e-06 * tc[2]
            +1.74496729e-10 * tc[3]
            -1.18021041e-14 * tc[4]
            +2.27163806e+00 ;
        /*species 5: HCO */
        species[5] =
            +2.77217438e+00 * tc[0]
            +4.95695526e-03 * tc[1]
            -1.24222806e-06 * tc[2]
            +1.96387259e-10 * tc[3]
            -1.33377178e-14 * tc[4]
            +9.79834492e+00 ;
    }
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

