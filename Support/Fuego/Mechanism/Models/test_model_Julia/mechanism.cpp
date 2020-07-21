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

    double fwd_A_DEF[10], fwd_beta_DEF[10], fwd_Ea_DEF[10];
    double low_A_DEF[10], low_beta_DEF[10], low_Ea_DEF[10];
    double rev_A_DEF[10], rev_beta_DEF[10], rev_Ea_DEF[10];
    double troe_a_DEF[10],troe_Ts_DEF[10], troe_Tss_DEF[10], troe_Tsss_DEF[10];
    double sri_a_DEF[10], sri_b_DEF[10], sri_c_DEF[10], sri_d_DEF[10], sri_e_DEF[10];
    double activation_units_DEF[10], prefactor_units_DEF[10], phase_units_DEF[10];
    int is_PD_DEF[10], troe_len_DEF[10], sri_len_DEF[10], nTB_DEF[10], *TBid_DEF[10];
    double *TB_DEF[10];
    std::vector<int> rxn_map;
};

using namespace thermo;
#endif

/* Inverse molecular weights */
/* TODO: check necessity on CPU */
static AMREX_GPU_DEVICE_MANAGED double imw[14] = {
    1.0 / 2.015940,  /*H2 */
    1.0 / 1.007970,  /*H */
    1.0 / 15.999400,  /*O */
    1.0 / 31.998800,  /*O2 */
    1.0 / 17.007370,  /*OH */
    1.0 / 18.015340,  /*H2O */
    1.0 / 33.006770,  /*HO2 */
    1.0 / 34.014740,  /*H2O2 */
    1.0 / 12.011150,  /*C */
    1.0 / 13.019120,  /*CH */
    1.0 / 14.027090,  /*CH2 */
    1.0 / 28.010550,  /*CO */
    1.0 / 44.009950,  /*CO2 */
    1.0 / 29.018520};  /*HCO */

/* Inverse molecular weights */
/* TODO: check necessity because redundant with molecularWeight */
static AMREX_GPU_DEVICE_MANAGED double molecular_weights[14] = {
    2.015940,  /*H2 */
    1.007970,  /*H */
    15.999400,  /*O */
    31.998800,  /*O2 */
    17.007370,  /*OH */
    18.015340,  /*H2O */
    33.006770,  /*HO2 */
    34.014740,  /*H2O2 */
    12.011150,  /*C */
    13.019120,  /*CH */
    14.027090,  /*CH2 */
    28.010550,  /*CO */
    44.009950,  /*CO2 */
    29.018520};  /*HCO */

AMREX_GPU_HOST_DEVICE
void get_imw(double imw_new[]){
    for(int i = 0; i<14; ++i) imw_new[i] = imw[i];
}

/* TODO: check necessity because redundant with CKWT */
AMREX_GPU_HOST_DEVICE
void get_mw(double mw_new[]){
    for(int i = 0; i<14; ++i) mw_new[i] = molecular_weights[i];
}


#ifndef AMREX_USE_CUDA
/* Initializes parameter database */
void CKINIT()
{

    rxn_map = {0,1,2,3,4,5,6,7,8,9};

    // (0):  O + HO2 => OH + O2
    kiv[0] = {2,6,4,3};
    nuv[0] = {-1,-1,1,1};
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
    kiv[1] = {1,6,2,5};
    nuv[1] = {-1,-1,1,1};
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
    kiv[2] = {1,7,4,5};
    nuv[2] = {-1,-1,1,1};
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
    kiv[3] = {2,9,1,11};
    nuv[3] = {-1,-1,1,1};
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
    kiv[4] = {1,9,8,0};
    nuv[4] = {-1,-1,1,1};
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
    kiv[5] = {2,10,1,13};
    nuv[5] = {-1,-1,1,1};
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
    kiv[6] = {1,3,2,4};
    nuv[6] = {-1,-1,1,1};
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
    kiv[7] = {1,6,4};
    nuv[7] = {-1,-1,2.0};
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
    kiv[8] = {4,11,1,12};
    nuv[8] = {-1,-1,1,1};
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
    kiv[9] = {4,9,1,13};
    nuv[9] = {-1,-1,1,1};
    // (9):  OH + CH <=> H + HCO
    fwd_A[9]     = 30000000000000;
    fwd_beta[9]  = 0;
    fwd_Ea[9]    = 0;
    prefactor_units[9]  = 1.0000000000000002e-06;
    activation_units[9] = 0.50321666580471969;
    phase_units[9]      = pow(10,-12.000000);
    is_PD[9] = 0;
    nTB[9] = 0;

    SetAllDefaults();
}

void GET_REACTION_MAP(int *rmap)
{
    for (int i=0; i<10; ++i) {
        rmap[i] = rxn_map[i] + 1;
    }
}

#include <ReactionData.H>
double* GetParamPtr(int                reaction_id,
                    REACTION_PARAMETER param_id,
                    int                species_id,
                    int                get_default)
{
  double* ret = 0;
  if (reaction_id<0 || reaction_id>=10) {
    printf("Bad reaction id = %d",reaction_id);
    abort();
  };
  int mrid = rxn_map[reaction_id];

  if (param_id == THIRD_BODY) {
    if (species_id<0 || species_id>=14) {
      printf("GetParamPtr: Bad species id = %d",species_id);
      abort();
    }
    if (get_default) {
      for (int i=0; i<nTB_DEF[mrid]; ++i) {
        if (species_id == TBid_DEF[mrid][i]) {
          ret = &(TB_DEF[mrid][i]);
        }
      }
    }
    else {
      for (int i=0; i<nTB[mrid]; ++i) {
        if (species_id == TBid[mrid][i]) {
          ret = &(TB[mrid][i]);
        }
      }
    }
    if (ret == 0) {
      printf("GetParamPtr: No TB for reaction id = %d",reaction_id);
      abort();
    }
  }
  else {
    if (     param_id == FWD_A)     {ret = (get_default ? &(fwd_A_DEF[mrid]) : &(fwd_A[mrid]));}
      else if (param_id == FWD_BETA)  {ret = (get_default ? &(fwd_beta_DEF[mrid]) : &(fwd_beta[mrid]));}
      else if (param_id == FWD_EA)    {ret = (get_default ? &(fwd_Ea_DEF[mrid]) : &(fwd_Ea[mrid]));}
      else if (param_id == LOW_A)     {ret = (get_default ? &(low_A_DEF[mrid]) : &(low_A[mrid]));}
      else if (param_id == LOW_BETA)  {ret = (get_default ? &(low_beta_DEF[mrid]) : &(low_beta[mrid]));}
      else if (param_id == LOW_EA)    {ret = (get_default ? &(low_Ea_DEF[mrid]) : &(low_Ea[mrid]));}
      else if (param_id == REV_A)     {ret = (get_default ? &(rev_A_DEF[mrid]) : &(rev_A[mrid]));}
      else if (param_id == REV_BETA)  {ret = (get_default ? &(rev_beta_DEF[mrid]) : &(rev_beta[mrid]));}
      else if (param_id == REV_EA)    {ret = (get_default ? &(rev_Ea_DEF[mrid]) : &(rev_Ea[mrid]));}
      else if (param_id == TROE_A)    {ret = (get_default ? &(troe_a_DEF[mrid]) : &(troe_a[mrid]));}
      else if (param_id == TROE_TS)   {ret = (get_default ? &(troe_Ts_DEF[mrid]) : &(troe_Ts[mrid]));}
      else if (param_id == TROE_TSS)  {ret = (get_default ? &(troe_Tss_DEF[mrid]) : &(troe_Tss[mrid]));}
      else if (param_id == TROE_TSSS) {ret = (get_default ? &(troe_Tsss_DEF[mrid]) : &(troe_Tsss[mrid]));}
      else if (param_id == SRI_A)     {ret = (get_default ? &(sri_a_DEF[mrid]) : &(sri_a[mrid]));}
      else if (param_id == SRI_B)     {ret = (get_default ? &(sri_b_DEF[mrid]) : &(sri_b[mrid]));}
      else if (param_id == SRI_C)     {ret = (get_default ? &(sri_c_DEF[mrid]) : &(sri_c[mrid]));}
      else if (param_id == SRI_D)     {ret = (get_default ? &(sri_d_DEF[mrid]) : &(sri_d[mrid]));}
      else if (param_id == SRI_E)     {ret = (get_default ? &(sri_e_DEF[mrid]) : &(sri_e[mrid]));}
    else {
      printf("GetParamPtr: Unknown parameter id");
      abort();
    }
  }
  return ret;
}

void ResetAllParametersToDefault()
{
    for (int i=0; i<10; i++) {
        if (nTB[i] != 0) {
            nTB[i] = 0;
            free(TB[i]);
            free(TBid[i]);
        }

        fwd_A[i]    = fwd_A_DEF[i];
        fwd_beta[i] = fwd_beta_DEF[i];
        fwd_Ea[i]   = fwd_Ea_DEF[i];

        low_A[i]    = low_A_DEF[i];
        low_beta[i] = low_beta_DEF[i];
        low_Ea[i]   = low_Ea_DEF[i];

        rev_A[i]    = rev_A_DEF[i];
        rev_beta[i] = rev_beta_DEF[i];
        rev_Ea[i]   = rev_Ea_DEF[i];

        troe_a[i]    = troe_a_DEF[i];
        troe_Ts[i]   = troe_Ts_DEF[i];
        troe_Tss[i]  = troe_Tss_DEF[i];
        troe_Tsss[i] = troe_Tsss_DEF[i];

        sri_a[i] = sri_a_DEF[i];
        sri_b[i] = sri_b_DEF[i];
        sri_c[i] = sri_c_DEF[i];
        sri_d[i] = sri_d_DEF[i];
        sri_e[i] = sri_e_DEF[i];

        is_PD[i]    = is_PD_DEF[i];
        troe_len[i] = troe_len_DEF[i];
        sri_len[i]  = sri_len_DEF[i];

        activation_units[i] = activation_units_DEF[i];
        prefactor_units[i]  = prefactor_units_DEF[i];
        phase_units[i]      = phase_units_DEF[i];

        nTB[i]  = nTB_DEF[i];
        if (nTB[i] != 0) {
           TB[i] = (double *) malloc(sizeof(double) * nTB[i]);
           TBid[i] = (int *) malloc(sizeof(int) * nTB[i]);
           for (int j=0; j<nTB[i]; j++) {
             TB[i][j] = TB_DEF[i][j];
             TBid[i][j] = TBid_DEF[i][j];
           }
        }
    }
}

void SetAllDefaults()
{
    for (int i=0; i<10; i++) {
        if (nTB_DEF[i] != 0) {
            nTB_DEF[i] = 0;
            free(TB_DEF[i]);
            free(TBid_DEF[i]);
        }

        fwd_A_DEF[i]    = fwd_A[i];
        fwd_beta_DEF[i] = fwd_beta[i];
        fwd_Ea_DEF[i]   = fwd_Ea[i];

        low_A_DEF[i]    = low_A[i];
        low_beta_DEF[i] = low_beta[i];
        low_Ea_DEF[i]   = low_Ea[i];

        rev_A_DEF[i]    = rev_A[i];
        rev_beta_DEF[i] = rev_beta[i];
        rev_Ea_DEF[i]   = rev_Ea[i];

        troe_a_DEF[i]    = troe_a[i];
        troe_Ts_DEF[i]   = troe_Ts[i];
        troe_Tss_DEF[i]  = troe_Tss[i];
        troe_Tsss_DEF[i] = troe_Tsss[i];

        sri_a_DEF[i] = sri_a[i];
        sri_b_DEF[i] = sri_b[i];
        sri_c_DEF[i] = sri_c[i];
        sri_d_DEF[i] = sri_d[i];
        sri_e_DEF[i] = sri_e[i];

        is_PD_DEF[i]    = is_PD[i];
        troe_len_DEF[i] = troe_len[i];
        sri_len_DEF[i]  = sri_len[i];

        activation_units_DEF[i] = activation_units[i];
        prefactor_units_DEF[i]  = prefactor_units[i];
        phase_units_DEF[i]      = phase_units[i];

        nTB_DEF[i]  = nTB[i];
        if (nTB_DEF[i] != 0) {
           TB_DEF[i] = (double *) malloc(sizeof(double) * nTB_DEF[i]);
           TBid_DEF[i] = (int *) malloc(sizeof(int) * nTB_DEF[i]);
           for (int j=0; j<nTB_DEF[i]; j++) {
             TB_DEF[i][j] = TB[i][j];
             TBid_DEF[i][j] = TBid[i][j];
           }
        }
    }
}

/* Finalizes parameter database */
void CKFINALIZE()
{
  for (int i=0; i<10; ++i) {
    free(TB[i]); TB[i] = 0; 
    free(TBid[i]); TBid[i] = 0;
    nTB[i] = 0;

    free(TB_DEF[i]); TB_DEF[i] = 0; 
    free(TBid_DEF[i]); TBid_DEF[i] = 0;
    nTB_DEF[i] = 0;
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
    *kk = 14;
    *ii = 10;
    *nfit = -1; /*Why do you need this anyway ?  */
}



/* ckxnum... for parsing strings  */
void CKXNUM(char * line, int * nexp, int * lout, int * nval, double *  rval, int * kerr, int lenline )
{
    int n,i; /*Loop Counters */
    char cstr[1000];
    char *saveptr;
    char *p; /*String Tokens */
    /* Strip Comments  */
    for (i=0; i<lenline; ++i) {
        if (line[i]=='!') {
            break;
        }
        cstr[i] = line[i];
    }
    cstr[i] = '\0';

    p = strtok_r(cstr," ", &saveptr);
    if (!p) {
        *nval = 0;
        *kerr = 1;
        return;
    }
    for (n=0; n<*nexp; ++n) {
        rval[n] = atof(p);
        p = strtok_r(NULL, " ", &saveptr);
        if (!p) break;
    }
    *nval = n+1;
    if (*nval < *nexp) *kerr = 1;
    return;
}


/* cksnum... for parsing strings  */
void CKSNUM(char * line, int * nexp, int * lout, char * kray, int * nn, int * knum, int * nval, double *  rval, int * kerr, int lenline, int lenkray)
{
    /*Not done yet ... */
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
    kname.resize(14);
    kname[0] = "H2";
    kname[1] = "H";
    kname[2] = "O";
    kname[3] = "O2";
    kname[4] = "OH";
    kname[5] = "H2O";
    kname[6] = "HO2";
    kname[7] = "H2O2";
    kname[8] = "C";
    kname[9] = "CH";
    kname[10] = "CH2";
    kname[11] = "CO";
    kname[12] = "CO2";
    kname[13] = "HCO";
}


/* Returns the char strings of species names */
void CKSYMS(int * kname, int * plenkname )
{
    int i; /*Loop Counter */
    int lenkname = *plenkname;
    /*clear kname */
    for (i=0; i<lenkname*14; i++) {
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

    /* O2  */
    kname[ 3*lenkname + 0 ] = 'O';
    kname[ 3*lenkname + 1 ] = '2';
    kname[ 3*lenkname + 2 ] = ' ';

    /* OH  */
    kname[ 4*lenkname + 0 ] = 'O';
    kname[ 4*lenkname + 1 ] = 'H';
    kname[ 4*lenkname + 2 ] = ' ';

    /* H2O  */
    kname[ 5*lenkname + 0 ] = 'H';
    kname[ 5*lenkname + 1 ] = '2';
    kname[ 5*lenkname + 2 ] = 'O';
    kname[ 5*lenkname + 3 ] = ' ';

    /* HO2  */
    kname[ 6*lenkname + 0 ] = 'H';
    kname[ 6*lenkname + 1 ] = 'O';
    kname[ 6*lenkname + 2 ] = '2';
    kname[ 6*lenkname + 3 ] = ' ';

    /* H2O2  */
    kname[ 7*lenkname + 0 ] = 'H';
    kname[ 7*lenkname + 1 ] = '2';
    kname[ 7*lenkname + 2 ] = 'O';
    kname[ 7*lenkname + 3 ] = '2';
    kname[ 7*lenkname + 4 ] = ' ';

    /* C  */
    kname[ 8*lenkname + 0 ] = 'C';
    kname[ 8*lenkname + 1 ] = ' ';

    /* CH  */
    kname[ 9*lenkname + 0 ] = 'C';
    kname[ 9*lenkname + 1 ] = 'H';
    kname[ 9*lenkname + 2 ] = ' ';

    /* CH2  */
    kname[ 10*lenkname + 0 ] = 'C';
    kname[ 10*lenkname + 1 ] = 'H';
    kname[ 10*lenkname + 2 ] = '2';
    kname[ 10*lenkname + 3 ] = ' ';

    /* CO  */
    kname[ 11*lenkname + 0 ] = 'C';
    kname[ 11*lenkname + 1 ] = 'O';
    kname[ 11*lenkname + 2 ] = ' ';

    /* CO2  */
    kname[ 12*lenkname + 0 ] = 'C';
    kname[ 12*lenkname + 1 ] = 'O';
    kname[ 12*lenkname + 2 ] = '2';
    kname[ 12*lenkname + 3 ] = ' ';

    /* HCO  */
    kname[ 13*lenkname + 0 ] = 'H';
    kname[ 13*lenkname + 1 ] = 'C';
    kname[ 13*lenkname + 2 ] = 'O';
    kname[ 13*lenkname + 3 ] = ' ';

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
    XW += x[3]*31.998800; /*O2 */
    XW += x[4]*17.007370; /*OH */
    XW += x[5]*18.015340; /*H2O */
    XW += x[6]*33.006770; /*HO2 */
    XW += x[7]*34.014740; /*H2O2 */
    XW += x[8]*12.011150; /*C */
    XW += x[9]*13.019120; /*CH */
    XW += x[10]*14.027090; /*CH2 */
    XW += x[11]*28.010550; /*CO */
    XW += x[12]*44.009950; /*CO2 */
    XW += x[13]*29.018520; /*HCO */
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
    YOW += y[3]*imw[3]; /*O2 */
    YOW += y[4]*imw[4]; /*OH */
    YOW += y[5]*imw[5]; /*H2O */
    YOW += y[6]*imw[6]; /*HO2 */
    YOW += y[7]*imw[7]; /*H2O2 */
    YOW += y[8]*imw[8]; /*C */
    YOW += y[9]*imw[9]; /*CH */
    YOW += y[10]*imw[10]; /*CH2 */
    YOW += y[11]*imw[11]; /*CO */
    YOW += y[12]*imw[12]; /*CO2 */
    YOW += y[13]*imw[13]; /*HCO */
    *P = *rho * 8.31446e+07 * (*T) * YOW; /*P = rho*R*T/W */

    return;
}


#ifndef AMREX_USE_CUDA
/*Compute P = rhoRT/W(y) */
void VCKPY(int *  np, double *  rho, double *  T, double *  y,  double *  P)
{
    double YOW[*np];
    for (int i=0; i<(*np); i++) {
        YOW[i] = 0.0;
    }

    for (int n=0; n<14; n++) {
        for (int i=0; i<(*np); i++) {
            YOW[i] += y[n*(*np)+i] * imw[n];
        }
    }

    for (int i=0; i<(*np); i++) {
        P[i] = rho[i] * 8.31446e+07 * T[i] * YOW[i]; /*P = rho*R*T/W */
    }

    return;
}
#endif


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
    W += c[3]*31.998800; /*O2 */
    W += c[4]*17.007370; /*OH */
    W += c[5]*18.015340; /*H2O */
    W += c[6]*33.006770; /*HO2 */
    W += c[7]*34.014740; /*H2O2 */
    W += c[8]*12.011150; /*C */
    W += c[9]*13.019120; /*CH */
    W += c[10]*14.027090; /*CH2 */
    W += c[11]*28.010550; /*CO */
    W += c[12]*44.009950; /*CO2 */
    W += c[13]*29.018520; /*HCO */

    for (id = 0; id < 14; ++id) {
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
    XW += x[3]*31.998800; /*O2 */
    XW += x[4]*17.007370; /*OH */
    XW += x[5]*18.015340; /*H2O */
    XW += x[6]*33.006770; /*HO2 */
    XW += x[7]*34.014740; /*H2O2 */
    XW += x[8]*12.011150; /*C */
    XW += x[9]*13.019120; /*CH */
    XW += x[10]*14.027090; /*CH2 */
    XW += x[11]*28.010550; /*CO */
    XW += x[12]*44.009950; /*CO2 */
    XW += x[13]*29.018520; /*HCO */
    *rho = *P * XW / (8.31446e+07 * (*T)); /*rho = P*W/(R*T) */

    return;
}


/*Compute rho = P*W(y)/RT */
AMREX_GPU_HOST_DEVICE void CKRHOY(double *  P, double *  T, double *  y,  double *  rho)
{
    double YOW = 0;
    double tmp[14];

    for (int i = 0; i < 14; i++)
    {
        tmp[i] = y[i]*imw[i];
    }
    for (int i = 0; i < 14; i++)
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
    W += c[3]*31.998800; /*O2 */
    W += c[4]*17.007370; /*OH */
    W += c[5]*18.015340; /*H2O */
    W += c[6]*33.006770; /*HO2 */
    W += c[7]*34.014740; /*H2O2 */
    W += c[8]*12.011150; /*C */
    W += c[9]*13.019120; /*CH */
    W += c[10]*14.027090; /*CH2 */
    W += c[11]*28.010550; /*CO */
    W += c[12]*44.009950; /*CO2 */
    W += c[13]*29.018520; /*HCO */

    for (id = 0; id < 14; ++id) {
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
    double tmp[14];

    for (int i = 0; i < 14; i++)
    {
        tmp[i] = y[i]*imw[i];
    }
    for (int i = 0; i < 14; i++)
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
    XW += x[3]*31.998800; /*O2 */
    XW += x[4]*17.007370; /*OH */
    XW += x[5]*18.015340; /*H2O */
    XW += x[6]*33.006770; /*HO2 */
    XW += x[7]*34.014740; /*H2O2 */
    XW += x[8]*12.011150; /*C */
    XW += x[9]*13.019120; /*CH */
    XW += x[10]*14.027090; /*CH2 */
    XW += x[11]*28.010550; /*CO */
    XW += x[12]*44.009950; /*CO2 */
    XW += x[13]*29.018520; /*HCO */
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
    W += c[3]*31.998800; /*O2 */
    W += c[4]*17.007370; /*OH */
    W += c[5]*18.015340; /*H2O */
    W += c[6]*33.006770; /*HO2 */
    W += c[7]*34.014740; /*H2O2 */
    W += c[8]*12.011150; /*C */
    W += c[9]*13.019120; /*CH */
    W += c[10]*14.027090; /*CH2 */
    W += c[11]*28.010550; /*CO */
    W += c[12]*44.009950; /*CO2 */
    W += c[13]*29.018520; /*HCO */

    for (id = 0; id < 14; ++id) {
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
    double tmp[14];

    for (int i = 0; i < 14; i++)
    {
        tmp[i] = y[i]*imw[i];
    }
    for (int i = 0; i < 14; i++)
    {
        YOW += tmp[i];
    }

    double YOWINV = 1.0/YOW;

    for (int i = 0; i < 14; i++)
    {
        x[i] = y[i]*imw[i]*YOWINV;
    }
    return;
}


#ifndef AMREX_USE_CUDA
/*convert y[npoints*species] (mass fracs) to x[npoints*species] (mole fracs) */
void VCKYTX(int *  np, double *  y,  double *  x)
{
    double YOW[*np];
    for (int i=0; i<(*np); i++) {
        YOW[i] = 0.0;
    }

    for (int n=0; n<14; n++) {
        for (int i=0; i<(*np); i++) {
            x[n*(*np)+i] = y[n*(*np)+i] * imw[n];
            YOW[i] += x[n*(*np)+i];
        }
    }

    for (int i=0; i<(*np); i++) {
        YOW[i] = 1.0/YOW[i];
    }

    for (int n=0; n<14; n++) {
        for (int i=0; i<(*np); i++) {
            x[n*(*np)+i] *=  YOW[i];
        }
    }
}
#else
/*TODO: remove this on GPU */
void VCKYTX(int *  np, double *  y,  double *  x)
{
}
#endif


/*convert y[species] (mass fracs) to c[species] (molar conc) */
void CKYTCP(double *  P, double *  T, double *  y,  double *  c)
{
    double YOW = 0;
    double PWORT;

    /*Compute inverse of mean molecular wt first */
    for (int i = 0; i < 14; i++)
    {
        c[i] = y[i]*imw[i];
    }
    for (int i = 0; i < 14; i++)
    {
        YOW += c[i];
    }

    /*PW/RT (see Eq. 7) */
    PWORT = (*P)/(YOW * 8.31446e+07 * (*T)); 
    /*Now compute conversion */

    for (int i = 0; i < 14; i++)
    {
        c[i] = PWORT * y[i] * imw[i];
    }
    return;
}


/*convert y[species] (mass fracs) to c[species] (molar conc) */
AMREX_GPU_HOST_DEVICE void CKYTCR(double *  rho, double *  T, double *  y,  double *  c)
{
    for (int i = 0; i < 14; i++)
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
    XW += x[3]*31.998800; /*O2 */
    XW += x[4]*17.007370; /*OH */
    XW += x[5]*18.015340; /*H2O */
    XW += x[6]*33.006770; /*HO2 */
    XW += x[7]*34.014740; /*H2O2 */
    XW += x[8]*12.011150; /*C */
    XW += x[9]*13.019120; /*CH */
    XW += x[10]*14.027090; /*CH2 */
    XW += x[11]*28.010550; /*CO */
    XW += x[12]*44.009950; /*CO2 */
    XW += x[13]*29.018520; /*HCO */
    /*Now compute conversion */
    double XWinv = 1.0/XW;
    y[0] = x[0]*2.015940*XWinv; 
    y[1] = x[1]*1.007970*XWinv; 
    y[2] = x[2]*15.999400*XWinv; 
    y[3] = x[3]*31.998800*XWinv; 
    y[4] = x[4]*17.007370*XWinv; 
    y[5] = x[5]*18.015340*XWinv; 
    y[6] = x[6]*33.006770*XWinv; 
    y[7] = x[7]*34.014740*XWinv; 
    y[8] = x[8]*12.011150*XWinv; 
    y[9] = x[9]*13.019120*XWinv; 
    y[10] = x[10]*14.027090*XWinv; 
    y[11] = x[11]*28.010550*XWinv; 
    y[12] = x[12]*44.009950*XWinv; 
    y[13] = x[13]*29.018520*XWinv; 

    return;
}


/*convert x[species] (mole fracs) to c[species] (molar conc) */
void CKXTCP(double *  P, double *  T, double *  x,  double *  c)
{
    int id; /*loop counter */
    double PORT = (*P)/(8.31446e+07 * (*T)); /*P/RT */

    /*Compute conversion, see Eq 10 */
    for (id = 0; id < 14; ++id) {
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
    XW += x[3]*31.998800; /*O2 */
    XW += x[4]*17.007370; /*OH */
    XW += x[5]*18.015340; /*H2O */
    XW += x[6]*33.006770; /*HO2 */
    XW += x[7]*34.014740; /*H2O2 */
    XW += x[8]*12.011150; /*C */
    XW += x[9]*13.019120; /*CH */
    XW += x[10]*14.027090; /*CH2 */
    XW += x[11]*28.010550; /*CO */
    XW += x[12]*44.009950; /*CO2 */
    XW += x[13]*29.018520; /*HCO */
    ROW = (*rho) / XW;

    /*Compute conversion, see Eq 11 */
    for (id = 0; id < 14; ++id) {
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
    for (id = 0; id < 14; ++id) {
        sumC += c[id];
    }

    /* See Eq 13  */
    double sumCinv = 1.0/sumC;
    for (id = 0; id < 14; ++id) {
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
    CW += c[3]*31.998800; /*O2 */
    CW += c[4]*17.007370; /*OH */
    CW += c[5]*18.015340; /*H2O */
    CW += c[6]*33.006770; /*HO2 */
    CW += c[7]*34.014740; /*H2O2 */
    CW += c[8]*12.011150; /*C */
    CW += c[9]*13.019120; /*CH */
    CW += c[10]*14.027090; /*CH2 */
    CW += c[11]*28.010550; /*CO */
    CW += c[12]*44.009950; /*CO2 */
    CW += c[13]*29.018520; /*HCO */
    /*Now compute conversion */
    double CWinv = 1.0/CW;
    y[0] = c[0]*2.015940*CWinv; 
    y[1] = c[1]*1.007970*CWinv; 
    y[2] = c[2]*15.999400*CWinv; 
    y[3] = c[3]*31.998800*CWinv; 
    y[4] = c[4]*17.007370*CWinv; 
    y[5] = c[5]*18.015340*CWinv; 
    y[6] = c[6]*33.006770*CWinv; 
    y[7] = c[7]*34.014740*CWinv; 
    y[8] = c[8]*12.011150*CWinv; 
    y[9] = c[9]*13.019120*CWinv; 
    y[10] = c[10]*14.027090*CWinv; 
    y[11] = c[11]*28.010550*CWinv; 
    y[12] = c[12]*44.009950*CWinv; 
    y[13] = c[13]*29.018520*CWinv; 

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


/*get specific heat at constant volume as a function  */
/*of T for all species (molar units) */
void CKCVML(double *  T,  double *  cvml)
{
    int id; /*loop counter */
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    cv_R(cvml, tc);

    /*convert to chemkin units */
    for (id = 0; id < 14; ++id) {
        cvml[id] *= 8.31446e+07;
    }
}


/*get specific heat at constant pressure as a  */
/*function of T for all species (molar units) */
void CKCPML(double *  T,  double *  cpml)
{
    int id; /*loop counter */
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    cp_R(cpml, tc);

    /*convert to chemkin units */
    for (id = 0; id < 14; ++id) {
        cpml[id] *= 8.31446e+07;
    }
}


/*get internal energy as a function  */
/*of T for all species (molar units) */
void CKUML(double *  T,  double *  uml)
{
    int id; /*loop counter */
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446e+07*tT; /*R*T */
    speciesInternalEnergy(uml, tc);

    /*convert to chemkin units */
    for (id = 0; id < 14; ++id) {
        uml[id] *= RT;
    }
}


/*get enthalpy as a function  */
/*of T for all species (molar units) */
void CKHML(double *  T,  double *  hml)
{
    int id; /*loop counter */
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446e+07*tT; /*R*T */
    speciesEnthalpy(hml, tc);

    /*convert to chemkin units */
    for (id = 0; id < 14; ++id) {
        hml[id] *= RT;
    }
}


/*get standard-state Gibbs energy as a function  */
/*of T for all species (molar units) */
void CKGML(double *  T,  double *  gml)
{
    int id; /*loop counter */
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446e+07*tT; /*R*T */
    gibbs(gml, tc);

    /*convert to chemkin units */
    for (id = 0; id < 14; ++id) {
        gml[id] *= RT;
    }
}


/*get standard-state Helmholtz free energy as a  */
/*function of T for all species (molar units) */
void CKAML(double *  T,  double *  aml)
{
    int id; /*loop counter */
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446e+07*tT; /*R*T */
    helmholtz(aml, tc);

    /*convert to chemkin units */
    for (id = 0; id < 14; ++id) {
        aml[id] *= RT;
    }
}


/*Returns the standard-state entropies in molar units */
void CKSML(double *  T,  double *  sml)
{
    int id; /*loop counter */
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    speciesEntropy(sml, tc);

    /*convert to chemkin units */
    for (id = 0; id < 14; ++id) {
        sml[id] *= 8.31446e+07;
    }
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
    cvms[3] *= 2.598367006935648e+06; /*O2 */
    cvms[4] *= 4.888740950630956e+06; /*OH */
    cvms[5] *= 4.615212712140454e+06; /*H2O */
    cvms[6] *= 2.519017346487778e+06; /*HO2 */
    cvms[7] *= 2.444370475315478e+06; /*H2O2 */
    cvms[8] *= 6.922286890225532e+06; /*C */
    cvms[9] *= 6.386347631908485e+06; /*CH */
    cvms[10] *= 5.927432288630956e+06; /*CH2 */
    cvms[11] *= 2.968332509769797e+06; /*CO */
    cvms[12] *= 1.889223372931176e+06; /*CO2 */
    cvms[13] *= 2.865226282440744e+06; /*HCO */
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
    cpms[3] *= 2.598367006935648e+06; /*O2 */
    cpms[4] *= 4.888740950630956e+06; /*OH */
    cpms[5] *= 4.615212712140454e+06; /*H2O */
    cpms[6] *= 2.519017346487778e+06; /*HO2 */
    cpms[7] *= 2.444370475315478e+06; /*H2O2 */
    cpms[8] *= 6.922286890225532e+06; /*C */
    cpms[9] *= 6.386347631908485e+06; /*CH */
    cpms[10] *= 5.927432288630956e+06; /*CH2 */
    cpms[11] *= 2.968332509769797e+06; /*CO */
    cpms[12] *= 1.889223372931176e+06; /*CO2 */
    cpms[13] *= 2.865226282440744e+06; /*HCO */
}


/*Returns internal energy in mass units (Eq 30.) */
AMREX_GPU_HOST_DEVICE void CKUMS(double *  T,  double *  ums)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446e+07*tT; /*R*T */
    speciesInternalEnergy(ums, tc);
    for (int i = 0; i < 14; i++)
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
    for (int i = 0; i < 14; i++)
    {
        hms[i] *= RT*imw[i];
    }
}


#ifndef AMREX_USE_CUDA
/*Returns enthalpy in mass units (Eq 27.) */
void VCKHMS(int *  np, double *  T,  double *  hms)
{
    double tc[5], h[14];

    for (int i=0; i<(*np); i++) {
        tc[0] = 0.0;
        tc[1] = T[i];
        tc[2] = T[i]*T[i];
        tc[3] = T[i]*T[i]*T[i];
        tc[4] = T[i]*T[i]*T[i]*T[i];

        speciesEnthalpy(h, tc);

        hms[0*(*np)+i] = h[0];
        hms[1*(*np)+i] = h[1];
        hms[2*(*np)+i] = h[2];
        hms[3*(*np)+i] = h[3];
        hms[4*(*np)+i] = h[4];
        hms[5*(*np)+i] = h[5];
        hms[6*(*np)+i] = h[6];
        hms[7*(*np)+i] = h[7];
        hms[8*(*np)+i] = h[8];
        hms[9*(*np)+i] = h[9];
        hms[10*(*np)+i] = h[10];
        hms[11*(*np)+i] = h[11];
        hms[12*(*np)+i] = h[12];
        hms[13*(*np)+i] = h[13];
    }

    for (int n=0; n<14; n++) {
        for (int i=0; i<(*np); i++) {
            hms[n*(*np)+i] *= 8.31446e+07 * T[i] * imw[n];
        }
    }
}
#else
/*TODO: remove this on GPU */
void VCKHMS(int *  np, double *  T,  double *  hms)
{
}
#endif


/*Returns gibbs in mass units (Eq 31.) */
void CKGMS(double *  T,  double *  gms)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446e+07*tT; /*R*T */
    gibbs(gms, tc);
    for (int i = 0; i < 14; i++)
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
    for (int i = 0; i < 14; i++)
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
    sms[3] *= 2.598367006935648e+06; /*O2 */
    sms[4] *= 4.888740950630956e+06; /*OH */
    sms[5] *= 4.615212712140454e+06; /*H2O */
    sms[6] *= 2.519017346487778e+06; /*HO2 */
    sms[7] *= 2.444370475315478e+06; /*H2O2 */
    sms[8] *= 6.922286890225532e+06; /*C */
    sms[9] *= 6.386347631908485e+06; /*CH */
    sms[10] *= 5.927432288630956e+06; /*CH2 */
    sms[11] *= 2.968332509769797e+06; /*CO */
    sms[12] *= 1.889223372931176e+06; /*CO2 */
    sms[13] *= 2.865226282440744e+06; /*HCO */
}


/*Returns the mean specific heat at CP (Eq. 33) */
void CKCPBL(double *  T, double *  x,  double *  cpbl)
{
    int id; /*loop counter */
    double result = 0; 
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double cpor[14]; /* temporary storage */
    cp_R(cpor, tc);

    /*perform dot product */
    for (id = 0; id < 14; ++id) {
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
    double cpor[14], tresult[14]; /* temporary storage */
    cp_R(cpor, tc);
    for (int i = 0; i < 14; i++)
    {
        tresult[i] = cpor[i]*y[i]*imw[i];

    }
    for (int i = 0; i < 14; i++)
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
    double cvor[14]; /* temporary storage */
    cv_R(cvor, tc);

    /*perform dot product */
    for (id = 0; id < 14; ++id) {
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
    double cvor[14]; /* temporary storage */
    cv_R(cvor, tc);
    /*multiply by y/molecularweight */
    result += cvor[0]*y[0]*imw[0]; /*H2 */
    result += cvor[1]*y[1]*imw[1]; /*H */
    result += cvor[2]*y[2]*imw[2]; /*O */
    result += cvor[3]*y[3]*imw[3]; /*O2 */
    result += cvor[4]*y[4]*imw[4]; /*OH */
    result += cvor[5]*y[5]*imw[5]; /*H2O */
    result += cvor[6]*y[6]*imw[6]; /*HO2 */
    result += cvor[7]*y[7]*imw[7]; /*H2O2 */
    result += cvor[8]*y[8]*imw[8]; /*C */
    result += cvor[9]*y[9]*imw[9]; /*CH */
    result += cvor[10]*y[10]*imw[10]; /*CH2 */
    result += cvor[11]*y[11]*imw[11]; /*CO */
    result += cvor[12]*y[12]*imw[12]; /*CO2 */
    result += cvor[13]*y[13]*imw[13]; /*HCO */

    *cvbs = result * 8.31446e+07;
}


/*Returns the mean enthalpy of the mixture in molar units */
void CKHBML(double *  T, double *  x,  double *  hbml)
{
    int id; /*loop counter */
    double result = 0; 
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double hml[14]; /* temporary storage */
    double RT = 8.31446e+07*tT; /*R*T */
    speciesEnthalpy(hml, tc);

    /*perform dot product */
    for (id = 0; id < 14; ++id) {
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
    double hml[14], tmp[14]; /* temporary storage */
    double RT = 8.31446e+07*tT; /*R*T */
    speciesEnthalpy(hml, tc);
    int id;
    for (id = 0; id < 14; ++id) {
        tmp[id] = y[id]*hml[id]*imw[id];
    }
    for (id = 0; id < 14; ++id) {
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
    double uml[14]; /* temporary energy array */
    double RT = 8.31446e+07*tT; /*R*T */
    speciesInternalEnergy(uml, tc);

    /*perform dot product */
    for (id = 0; id < 14; ++id) {
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
    double ums[14]; /* temporary energy array */
    double RT = 8.31446e+07*tT; /*R*T */
    speciesInternalEnergy(ums, tc);
    /*perform dot product + scaling by wt */
    result += y[0]*ums[0]*imw[0]; /*H2 */
    result += y[1]*ums[1]*imw[1]; /*H */
    result += y[2]*ums[2]*imw[2]; /*O */
    result += y[3]*ums[3]*imw[3]; /*O2 */
    result += y[4]*ums[4]*imw[4]; /*OH */
    result += y[5]*ums[5]*imw[5]; /*H2O */
    result += y[6]*ums[6]*imw[6]; /*HO2 */
    result += y[7]*ums[7]*imw[7]; /*H2O2 */
    result += y[8]*ums[8]*imw[8]; /*C */
    result += y[9]*ums[9]*imw[9]; /*CH */
    result += y[10]*ums[10]*imw[10]; /*CH2 */
    result += y[11]*ums[11]*imw[11]; /*CO */
    result += y[12]*ums[12]*imw[12]; /*CO2 */
    result += y[13]*ums[13]*imw[13]; /*HCO */

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
    double sor[14]; /* temporary storage */
    speciesEntropy(sor, tc);

    /*Compute Eq 42 */
    for (id = 0; id < 14; ++id) {
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
    double sor[14]; /* temporary storage */
    double x[14]; /* need a ytx conversion */
    double YOW = 0; /*See Eq 4, 6 in CK Manual */
    /*Compute inverse of mean molecular wt first */
    YOW += y[0]*imw[0]; /*H2 */
    YOW += y[1]*imw[1]; /*H */
    YOW += y[2]*imw[2]; /*O */
    YOW += y[3]*imw[3]; /*O2 */
    YOW += y[4]*imw[4]; /*OH */
    YOW += y[5]*imw[5]; /*H2O */
    YOW += y[6]*imw[6]; /*HO2 */
    YOW += y[7]*imw[7]; /*H2O2 */
    YOW += y[8]*imw[8]; /*C */
    YOW += y[9]*imw[9]; /*CH */
    YOW += y[10]*imw[10]; /*CH2 */
    YOW += y[11]*imw[11]; /*CO */
    YOW += y[12]*imw[12]; /*CO2 */
    YOW += y[13]*imw[13]; /*HCO */
    /*Now compute y to x conversion */
    x[0] = y[0]/(2.015940*YOW); 
    x[1] = y[1]/(1.007970*YOW); 
    x[2] = y[2]/(15.999400*YOW); 
    x[3] = y[3]/(31.998800*YOW); 
    x[4] = y[4]/(17.007370*YOW); 
    x[5] = y[5]/(18.015340*YOW); 
    x[6] = y[6]/(33.006770*YOW); 
    x[7] = y[7]/(34.014740*YOW); 
    x[8] = y[8]/(12.011150*YOW); 
    x[9] = y[9]/(13.019120*YOW); 
    x[10] = y[10]/(14.027090*YOW); 
    x[11] = y[11]/(28.010550*YOW); 
    x[12] = y[12]/(44.009950*YOW); 
    x[13] = y[13]/(29.018520*YOW); 
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
    double gort[14]; /* temporary storage */
    /*Compute g/RT */
    gibbs(gort, tc);

    /*Compute Eq 44 */
    for (id = 0; id < 14; ++id) {
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
    double gort[14]; /* temporary storage */
    double x[14]; /* need a ytx conversion */
    double YOW = 0; /*To hold 1/molecularweight */
    /*Compute inverse of mean molecular wt first */
    YOW += y[0]*imw[0]; /*H2 */
    YOW += y[1]*imw[1]; /*H */
    YOW += y[2]*imw[2]; /*O */
    YOW += y[3]*imw[3]; /*O2 */
    YOW += y[4]*imw[4]; /*OH */
    YOW += y[5]*imw[5]; /*H2O */
    YOW += y[6]*imw[6]; /*HO2 */
    YOW += y[7]*imw[7]; /*H2O2 */
    YOW += y[8]*imw[8]; /*C */
    YOW += y[9]*imw[9]; /*CH */
    YOW += y[10]*imw[10]; /*CH2 */
    YOW += y[11]*imw[11]; /*CO */
    YOW += y[12]*imw[12]; /*CO2 */
    YOW += y[13]*imw[13]; /*HCO */
    /*Now compute y to x conversion */
    x[0] = y[0]/(2.015940*YOW); 
    x[1] = y[1]/(1.007970*YOW); 
    x[2] = y[2]/(15.999400*YOW); 
    x[3] = y[3]/(31.998800*YOW); 
    x[4] = y[4]/(17.007370*YOW); 
    x[5] = y[5]/(18.015340*YOW); 
    x[6] = y[6]/(33.006770*YOW); 
    x[7] = y[7]/(34.014740*YOW); 
    x[8] = y[8]/(12.011150*YOW); 
    x[9] = y[9]/(13.019120*YOW); 
    x[10] = y[10]/(14.027090*YOW); 
    x[11] = y[11]/(28.010550*YOW); 
    x[12] = y[12]/(44.009950*YOW); 
    x[13] = y[13]/(29.018520*YOW); 
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
    double aort[14]; /* temporary storage */
    /*Compute g/RT */
    helmholtz(aort, tc);

    /*Compute Eq 44 */
    for (id = 0; id < 14; ++id) {
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
    double aort[14]; /* temporary storage */
    double x[14]; /* need a ytx conversion */
    double YOW = 0; /*To hold 1/molecularweight */
    /*Compute inverse of mean molecular wt first */
    YOW += y[0]*imw[0]; /*H2 */
    YOW += y[1]*imw[1]; /*H */
    YOW += y[2]*imw[2]; /*O */
    YOW += y[3]*imw[3]; /*O2 */
    YOW += y[4]*imw[4]; /*OH */
    YOW += y[5]*imw[5]; /*H2O */
    YOW += y[6]*imw[6]; /*HO2 */
    YOW += y[7]*imw[7]; /*H2O2 */
    YOW += y[8]*imw[8]; /*C */
    YOW += y[9]*imw[9]; /*CH */
    YOW += y[10]*imw[10]; /*CH2 */
    YOW += y[11]*imw[11]; /*CO */
    YOW += y[12]*imw[12]; /*CO2 */
    YOW += y[13]*imw[13]; /*HCO */
    /*Now compute y to x conversion */
    x[0] = y[0]/(2.015940*YOW); 
    x[1] = y[1]/(1.007970*YOW); 
    x[2] = y[2]/(15.999400*YOW); 
    x[3] = y[3]/(31.998800*YOW); 
    x[4] = y[4]/(17.007370*YOW); 
    x[5] = y[5]/(18.015340*YOW); 
    x[6] = y[6]/(33.006770*YOW); 
    x[7] = y[7]/(34.014740*YOW); 
    x[8] = y[8]/(12.011150*YOW); 
    x[9] = y[9]/(13.019120*YOW); 
    x[10] = y[10]/(14.027090*YOW); 
    x[11] = y[11]/(28.010550*YOW); 
    x[12] = y[12]/(44.009950*YOW); 
    x[13] = y[13]/(29.018520*YOW); 
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
    /*Scale by RT/W */
    *abms = result * RT * YOW;
}


/*compute the production rate for each species */
AMREX_GPU_HOST_DEVICE void CKWC(double *  T, double *  C,  double *  wdot)
{
    int id; /*loop counter */

    /*convert to SI */
    for (id = 0; id < 14; ++id) {
        C[id] *= 1.0e6;
    }

    /*convert to chemkin units */
    productionRate(wdot, C, *T);

    /*convert to chemkin units */
    for (id = 0; id < 14; ++id) {
        C[id] *= 1.0e-6;
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the molar production rate of species */
/*Given P, T, and mass fractions */
void CKWYP(double *  P, double *  T, double *  y,  double *  wdot)
{
    int id; /*loop counter */
    double c[14]; /*temporary storage */
    double YOW = 0; 
    double PWORT; 
    /*Compute inverse of mean molecular wt first */
    YOW += y[0]*imw[0]; /*H2 */
    YOW += y[1]*imw[1]; /*H */
    YOW += y[2]*imw[2]; /*O */
    YOW += y[3]*imw[3]; /*O2 */
    YOW += y[4]*imw[4]; /*OH */
    YOW += y[5]*imw[5]; /*H2O */
    YOW += y[6]*imw[6]; /*HO2 */
    YOW += y[7]*imw[7]; /*H2O2 */
    YOW += y[8]*imw[8]; /*C */
    YOW += y[9]*imw[9]; /*CH */
    YOW += y[10]*imw[10]; /*CH2 */
    YOW += y[11]*imw[11]; /*CO */
    YOW += y[12]*imw[12]; /*CO2 */
    YOW += y[13]*imw[13]; /*HCO */
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

    /*convert to chemkin units */
    productionRate(wdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 14; ++id) {
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the molar production rate of species */
/*Given P, T, and mole fractions */
void CKWXP(double *  P, double *  T, double *  x,  double *  wdot)
{
    int id; /*loop counter */
    double c[14]; /*temporary storage */
    double PORT = 1e6 * (*P)/(8.31446e+07 * (*T)); /*1e6 * P/RT so c goes to SI units */

    /*Compute conversion, see Eq 10 */
    for (id = 0; id < 14; ++id) {
        c[id] = x[id]*PORT;
    }

    /*convert to chemkin units */
    productionRate(wdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 14; ++id) {
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the molar production rate of species */
/*Given rho, T, and mass fractions */
AMREX_GPU_HOST_DEVICE void CKWYR(double *  rho, double *  T, double *  y,  double *  wdot)
{
    int id; /*loop counter */
    double c[14]; /*temporary storage */
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

    /*call productionRate */
    productionRate(wdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 14; ++id) {
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the molar production rate of species */
/*Given rho, T, and mass fractions */
void VCKWYR(int *  np, double *  rho, double *  T,
	    double *  y,
	    double *  wdot)
{
#ifndef AMREX_USE_CUDA
    double c[14*(*np)]; /*temporary storage */
    /*See Eq 8 with an extra 1e6 so c goes to SI */
    for (int n=0; n<14; n++) {
        for (int i=0; i<(*np); i++) {
            c[n*(*np)+i] = 1.0e6 * rho[i] * y[n*(*np)+i] * imw[n];
        }
    }

    /*call productionRate */
    vproductionRate(*np, wdot, c, T);

    /*convert to chemkin units */
    for (int i=0; i<14*(*np); i++) {
        wdot[i] *= 1.0e-6;
    }
#endif
}


/*Returns the molar production rate of species */
/*Given rho, T, and mole fractions */
void CKWXR(double *  rho, double *  T, double *  x,  double *  wdot)
{
    int id; /*loop counter */
    double c[14]; /*temporary storage */
    double XW = 0; /*See Eq 4, 11 in CK Manual */
    double ROW; 
    /*Compute mean molecular wt first */
    XW += x[0]*2.015940; /*H2 */
    XW += x[1]*1.007970; /*H */
    XW += x[2]*15.999400; /*O */
    XW += x[3]*31.998800; /*O2 */
    XW += x[4]*17.007370; /*OH */
    XW += x[5]*18.015340; /*H2O */
    XW += x[6]*33.006770; /*HO2 */
    XW += x[7]*34.014740; /*H2O2 */
    XW += x[8]*12.011150; /*C */
    XW += x[9]*13.019120; /*CH */
    XW += x[10]*14.027090; /*CH2 */
    XW += x[11]*28.010550; /*CO */
    XW += x[12]*44.009950; /*CO2 */
    XW += x[13]*29.018520; /*HCO */
    /*Extra 1e6 factor to take c to SI */
    ROW = 1e6*(*rho) / XW;

    /*Compute conversion, see Eq 11 */
    for (id = 0; id < 14; ++id) {
        c[id] = x[id]*ROW;
    }

    /*convert to chemkin units */
    productionRate(wdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 14; ++id) {
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the rate of progress for each reaction */
void CKQC(double *  T, double *  C, double *  qdot)
{
    int id; /*loop counter */

    /*convert to SI */
    for (id = 0; id < 14; ++id) {
        C[id] *= 1.0e6;
    }

    /*convert to chemkin units */
    progressRate(qdot, C, *T);

    /*convert to chemkin units */
    for (id = 0; id < 14; ++id) {
        C[id] *= 1.0e-6;
    }

    for (id = 0; id < 10; ++id) {
        qdot[id] *= 1.0e-6;
    }
}


/*Returns the progress rates of each reactions */
/*Given P, T, and mole fractions */
void CKKFKR(double *  P, double *  T, double *  x, double *  q_f, double *  q_r)
{
    int id; /*loop counter */
    double c[14]; /*temporary storage */
    double PORT = 1e6 * (*P)/(8.31446e+07 * (*T)); /*1e6 * P/RT so c goes to SI units */

    /*Compute conversion, see Eq 10 */
    for (id = 0; id < 14; ++id) {
        c[id] = x[id]*PORT;
    }

    /*convert to chemkin units */
    progressRateFR(q_f, q_r, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 10; ++id) {
        q_f[id] *= 1.0e-6;
        q_r[id] *= 1.0e-6;
    }
}


/*Returns the progress rates of each reactions */
/*Given P, T, and mass fractions */
void CKQYP(double *  P, double *  T, double *  y, double *  qdot)
{
    int id; /*loop counter */
    double c[14]; /*temporary storage */
    double YOW = 0; 
    double PWORT; 
    /*Compute inverse of mean molecular wt first */
    YOW += y[0]*imw[0]; /*H2 */
    YOW += y[1]*imw[1]; /*H */
    YOW += y[2]*imw[2]; /*O */
    YOW += y[3]*imw[3]; /*O2 */
    YOW += y[4]*imw[4]; /*OH */
    YOW += y[5]*imw[5]; /*H2O */
    YOW += y[6]*imw[6]; /*HO2 */
    YOW += y[7]*imw[7]; /*H2O2 */
    YOW += y[8]*imw[8]; /*C */
    YOW += y[9]*imw[9]; /*CH */
    YOW += y[10]*imw[10]; /*CH2 */
    YOW += y[11]*imw[11]; /*CO */
    YOW += y[12]*imw[12]; /*CO2 */
    YOW += y[13]*imw[13]; /*HCO */
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

    /*convert to chemkin units */
    progressRate(qdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 10; ++id) {
        qdot[id] *= 1.0e-6;
    }
}


/*Returns the progress rates of each reactions */
/*Given P, T, and mole fractions */
void CKQXP(double *  P, double *  T, double *  x, double *  qdot)
{
    int id; /*loop counter */
    double c[14]; /*temporary storage */
    double PORT = 1e6 * (*P)/(8.31446e+07 * (*T)); /*1e6 * P/RT so c goes to SI units */

    /*Compute conversion, see Eq 10 */
    for (id = 0; id < 14; ++id) {
        c[id] = x[id]*PORT;
    }

    /*convert to chemkin units */
    progressRate(qdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 10; ++id) {
        qdot[id] *= 1.0e-6;
    }
}


/*Returns the progress rates of each reactions */
/*Given rho, T, and mass fractions */
void CKQYR(double *  rho, double *  T, double *  y, double *  qdot)
{
    int id; /*loop counter */
    double c[14]; /*temporary storage */
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

    /*call progressRate */
    progressRate(qdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 10; ++id) {
        qdot[id] *= 1.0e-6;
    }
}


/*Returns the progress rates of each reactions */
/*Given rho, T, and mole fractions */
void CKQXR(double *  rho, double *  T, double *  x, double *  qdot)
{
    int id; /*loop counter */
    double c[14]; /*temporary storage */
    double XW = 0; /*See Eq 4, 11 in CK Manual */
    double ROW; 
    /*Compute mean molecular wt first */
    XW += x[0]*2.015940; /*H2 */
    XW += x[1]*1.007970; /*H */
    XW += x[2]*15.999400; /*O */
    XW += x[3]*31.998800; /*O2 */
    XW += x[4]*17.007370; /*OH */
    XW += x[5]*18.015340; /*H2O */
    XW += x[6]*33.006770; /*HO2 */
    XW += x[7]*34.014740; /*H2O2 */
    XW += x[8]*12.011150; /*C */
    XW += x[9]*13.019120; /*CH */
    XW += x[10]*14.027090; /*CH2 */
    XW += x[11]*28.010550; /*CO */
    XW += x[12]*44.009950; /*CO2 */
    XW += x[13]*29.018520; /*HCO */
    /*Extra 1e6 factor to take c to SI */
    ROW = 1e6*(*rho) / XW;

    /*Compute conversion, see Eq 11 */
    for (id = 0; id < 14; ++id) {
        c[id] = x[id]*ROW;
    }

    /*convert to chemkin units */
    progressRate(qdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 10; ++id) {
        qdot[id] *= 1.0e-6;
    }
}


/*Returns the stoichiometric coefficients */
/*of the reaction mechanism. (Eq 50) */
void CKNU(int * kdim,  int * nuki)
{
    int id; /*loop counter */
    int kd = (*kdim); 
    /*Zero nuki */
    for (id = 0; id < 14 * kd; ++ id) {
         nuki[id] = 0; 
    }

    /*reaction 1: O + HO2 => OH + O2 */
    nuki[ 2 * kd + 0 ] += -1.000000 ;
    nuki[ 6 * kd + 0 ] += -1.000000 ;
    nuki[ 4 * kd + 0 ] += +1.000000 ;
    nuki[ 3 * kd + 0 ] += +1.000000 ;

    /*reaction 2: H + HO2 => O + H2O */
    nuki[ 1 * kd + 1 ] += -1.000000 ;
    nuki[ 6 * kd + 1 ] += -1.000000 ;
    nuki[ 2 * kd + 1 ] += +1.000000 ;
    nuki[ 5 * kd + 1 ] += +1.000000 ;

    /*reaction 3: H + H2O2 <=> OH + H2O */
    nuki[ 1 * kd + 2 ] += -1.000000 ;
    nuki[ 7 * kd + 2 ] += -1.000000 ;
    nuki[ 4 * kd + 2 ] += +1.000000 ;
    nuki[ 5 * kd + 2 ] += +1.000000 ;

    /*reaction 4: O + CH => H + CO */
    nuki[ 2 * kd + 3 ] += -1.000000 ;
    nuki[ 9 * kd + 3 ] += -1.000000 ;
    nuki[ 1 * kd + 3 ] += +1.000000 ;
    nuki[ 11 * kd + 3 ] += +1.000000 ;

    /*reaction 5: H + CH <=> C + H2 */
    nuki[ 1 * kd + 4 ] += -1.000000 ;
    nuki[ 9 * kd + 4 ] += -1.000000 ;
    nuki[ 8 * kd + 4 ] += +1.000000 ;
    nuki[ 0 * kd + 4 ] += +1.000000 ;

    /*reaction 6: O + CH2 <=> H + HCO */
    nuki[ 2 * kd + 5 ] += -1.000000 ;
    nuki[ 10 * kd + 5 ] += -1.000000 ;
    nuki[ 1 * kd + 5 ] += +1.000000 ;
    nuki[ 13 * kd + 5 ] += +1.000000 ;

    /*reaction 7: H + O2 <=> O + OH */
    nuki[ 1 * kd + 6 ] += -1.000000 ;
    nuki[ 3 * kd + 6 ] += -1.000000 ;
    nuki[ 2 * kd + 6 ] += +1.000000 ;
    nuki[ 4 * kd + 6 ] += +1.000000 ;

    /*reaction 8: H + HO2 <=> 2.000000 OH */
    nuki[ 1 * kd + 7 ] += -1.000000 ;
    nuki[ 6 * kd + 7 ] += -1.000000 ;
    nuki[ 4 * kd + 7 ] += +2.000000 ;

    /*reaction 9: OH + CO <=> H + CO2 */
    nuki[ 4 * kd + 8 ] += -1.000000 ;
    nuki[ 11 * kd + 8 ] += -1.000000 ;
    nuki[ 1 * kd + 8 ] += +1.000000 ;
    nuki[ 12 * kd + 8 ] += +1.000000 ;

    /*reaction 10: OH + CH <=> H + HCO */
    nuki[ 4 * kd + 9 ] += -1.000000 ;
    nuki[ 9 * kd + 9 ] += -1.000000 ;
    nuki[ 1 * kd + 9 ] += +1.000000 ;
    nuki[ 13 * kd + 9 ] += +1.000000 ;
}


#ifndef AMREX_USE_CUDA
/*Returns a count of species in a reaction, and their indices */
/*and stoichiometric coefficients. (Eq 50) */
void CKINU(int * i, int * nspec, int * ki, int * nu)
{
    if (*i < 1) {
        /*Return max num species per reaction */
        *nspec = 4;
    } else {
        if (*i > 10) {
            *nspec = -1;
        } else {
            *nspec = kiv[*i-1].size();
            for (int j=0; j<*nspec; ++j) {
                ki[j] = kiv[*i-1][j] + 1;
                nu[j] = nuv[*i-1][j];
            }
        }
    }
}
#endif


/*Returns the elemental composition  */
/*of the speciesi (mdim is num of elements) */
void CKNCF(int * ncf)
{
    int id; /*loop counter */
    int kd = 3; 
    /*Zero ncf */
    for (id = 0; id < kd * 14; ++ id) {
         ncf[id] = 0; 
    }

    /*H2 */
    ncf[ 0 * kd + 1 ] = 2; /*H */

    /*H */
    ncf[ 1 * kd + 1 ] = 1; /*H */

    /*O */
    ncf[ 2 * kd + 0 ] = 1; /*O */

    /*O2 */
    ncf[ 3 * kd + 0 ] = 2; /*O */

    /*OH */
    ncf[ 4 * kd + 0 ] = 1; /*O */
    ncf[ 4 * kd + 1 ] = 1; /*H */

    /*H2O */
    ncf[ 5 * kd + 1 ] = 2; /*H */
    ncf[ 5 * kd + 0 ] = 1; /*O */

    /*HO2 */
    ncf[ 6 * kd + 1 ] = 1; /*H */
    ncf[ 6 * kd + 0 ] = 2; /*O */

    /*H2O2 */
    ncf[ 7 * kd + 1 ] = 2; /*H */
    ncf[ 7 * kd + 0 ] = 2; /*O */

    /*C */
    ncf[ 8 * kd + 2 ] = 1; /*C */

    /*CH */
    ncf[ 9 * kd + 2 ] = 1; /*C */
    ncf[ 9 * kd + 1 ] = 1; /*H */

    /*CH2 */
    ncf[ 10 * kd + 2 ] = 1; /*C */
    ncf[ 10 * kd + 1 ] = 2; /*H */

    /*CO */
    ncf[ 11 * kd + 2 ] = 1; /*C */
    ncf[ 11 * kd + 0 ] = 1; /*O */

    /*CO2 */
    ncf[ 12 * kd + 2 ] = 1; /*C */
    ncf[ 12 * kd + 0 ] = 2; /*O */

    /*HCO */
    ncf[ 13 * kd + 1 ] = 1; /*H */
    ncf[ 13 * kd + 2 ] = 1; /*C */
    ncf[ 13 * kd + 0 ] = 1; /*O */

}


/*Returns the arrehenius coefficients  */
/*for all reactions */
void CKABE( double *  a, double *  b, double *  e)
{
    // (0):  O + HO2 => OH + O2
    a[0] = 20000000000000;
    b[0] = 0;
    e[0] = 0;

    // (1):  H + HO2 => O + H2O
    a[1] = 3970000000000;
    b[1] = 0;
    e[1] = 671;

    // (2):  H + H2O2 <=> OH + H2O
    a[2] = 10000000000000;
    b[2] = 0;
    e[2] = 3600;

    // (3):  O + CH => H + CO
    a[3] = 57000000000000;
    b[3] = 0;
    e[3] = 0;

    // (4):  H + CH <=> C + H2
    a[4] = 110000000000000;
    b[4] = 0;
    e[4] = 0;

    // (5):  O + CH2 <=> H + HCO
    a[5] = 80000000000000;
    b[5] = 0;
    e[5] = 0;

    // (6):  H + O2 <=> O + OH
    a[6] = 83000000000000;
    b[6] = 0;
    e[6] = 14413;

    // (7):  H + HO2 <=> 2.000000 OH
    a[7] = 134000000000000;
    b[7] = 0;
    e[7] = 635;

    // (8):  OH + CO <=> H + CO2
    a[8] = 47600000;
    b[8] = 1.228;
    e[8] = 70;

    // (9):  OH + CH <=> H + HCO
    a[9] = 30000000000000;
    b[9] = 0;
    e[9] = 0;


    return;
}


/*Returns the equil constants for each reaction */
void CKEQC(double *  T, double *  C, double *  eqcon)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double gort[14]; /* temporary storage */

    /*compute the Gibbs free energy */
    gibbs(gort, tc);

    /*compute the equilibrium constants */
    equilibriumConstants(eqcon, gort, tT);

    /*reaction 1: O + HO2 => OH + O2 */
    /*eqcon[0] *= 1;  */

    /*reaction 2: H + HO2 => O + H2O */
    /*eqcon[1] *= 1;  */

    /*reaction 3: H + H2O2 <=> OH + H2O */
    /*eqcon[2] *= 1;  */

    /*reaction 4: O + CH => H + CO */
    /*eqcon[3] *= 1;  */

    /*reaction 5: H + CH <=> C + H2 */
    /*eqcon[4] *= 1;  */

    /*reaction 6: O + CH2 <=> H + HCO */
    /*eqcon[5] *= 1;  */

    /*reaction 7: H + O2 <=> O + OH */
    /*eqcon[6] *= 1;  */

    /*reaction 8: H + HO2 <=> 2.000000 OH */
    /*eqcon[7] *= 1;  */

    /*reaction 9: OH + CO <=> H + CO2 */
    /*eqcon[8] *= 1;  */

    /*reaction 10: OH + CH <=> H + HCO */
    /*eqcon[9] *= 1;  */
}


/*Returns the equil constants for each reaction */
/*Given P, T, and mass fractions */
void CKEQYP(double *  P, double *  T, double *  y, double *  eqcon)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double gort[14]; /* temporary storage */

    /*compute the Gibbs free energy */
    gibbs(gort, tc);

    /*compute the equilibrium constants */
    equilibriumConstants(eqcon, gort, tT);

    /*reaction 1: O + HO2 => OH + O2 */
    /*eqcon[0] *= 1;  */

    /*reaction 2: H + HO2 => O + H2O */
    /*eqcon[1] *= 1;  */

    /*reaction 3: H + H2O2 <=> OH + H2O */
    /*eqcon[2] *= 1;  */

    /*reaction 4: O + CH => H + CO */
    /*eqcon[3] *= 1;  */

    /*reaction 5: H + CH <=> C + H2 */
    /*eqcon[4] *= 1;  */

    /*reaction 6: O + CH2 <=> H + HCO */
    /*eqcon[5] *= 1;  */

    /*reaction 7: H + O2 <=> O + OH */
    /*eqcon[6] *= 1;  */

    /*reaction 8: H + HO2 <=> 2.000000 OH */
    /*eqcon[7] *= 1;  */

    /*reaction 9: OH + CO <=> H + CO2 */
    /*eqcon[8] *= 1;  */

    /*reaction 10: OH + CH <=> H + HCO */
    /*eqcon[9] *= 1;  */
}


/*Returns the equil constants for each reaction */
/*Given P, T, and mole fractions */
void CKEQXP(double *  P, double *  T, double *  x, double *  eqcon)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double gort[14]; /* temporary storage */

    /*compute the Gibbs free energy */
    gibbs(gort, tc);

    /*compute the equilibrium constants */
    equilibriumConstants(eqcon, gort, tT);

    /*reaction 1: O + HO2 => OH + O2 */
    /*eqcon[0] *= 1;  */

    /*reaction 2: H + HO2 => O + H2O */
    /*eqcon[1] *= 1;  */

    /*reaction 3: H + H2O2 <=> OH + H2O */
    /*eqcon[2] *= 1;  */

    /*reaction 4: O + CH => H + CO */
    /*eqcon[3] *= 1;  */

    /*reaction 5: H + CH <=> C + H2 */
    /*eqcon[4] *= 1;  */

    /*reaction 6: O + CH2 <=> H + HCO */
    /*eqcon[5] *= 1;  */

    /*reaction 7: H + O2 <=> O + OH */
    /*eqcon[6] *= 1;  */

    /*reaction 8: H + HO2 <=> 2.000000 OH */
    /*eqcon[7] *= 1;  */

    /*reaction 9: OH + CO <=> H + CO2 */
    /*eqcon[8] *= 1;  */

    /*reaction 10: OH + CH <=> H + HCO */
    /*eqcon[9] *= 1;  */
}


/*Returns the equil constants for each reaction */
/*Given rho, T, and mass fractions */
void CKEQYR(double *  rho, double *  T, double *  y, double *  eqcon)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double gort[14]; /* temporary storage */

    /*compute the Gibbs free energy */
    gibbs(gort, tc);

    /*compute the equilibrium constants */
    equilibriumConstants(eqcon, gort, tT);

    /*reaction 1: O + HO2 => OH + O2 */
    /*eqcon[0] *= 1;  */

    /*reaction 2: H + HO2 => O + H2O */
    /*eqcon[1] *= 1;  */

    /*reaction 3: H + H2O2 <=> OH + H2O */
    /*eqcon[2] *= 1;  */

    /*reaction 4: O + CH => H + CO */
    /*eqcon[3] *= 1;  */

    /*reaction 5: H + CH <=> C + H2 */
    /*eqcon[4] *= 1;  */

    /*reaction 6: O + CH2 <=> H + HCO */
    /*eqcon[5] *= 1;  */

    /*reaction 7: H + O2 <=> O + OH */
    /*eqcon[6] *= 1;  */

    /*reaction 8: H + HO2 <=> 2.000000 OH */
    /*eqcon[7] *= 1;  */

    /*reaction 9: OH + CO <=> H + CO2 */
    /*eqcon[8] *= 1;  */

    /*reaction 10: OH + CH <=> H + HCO */
    /*eqcon[9] *= 1;  */
}


/*Returns the equil constants for each reaction */
/*Given rho, T, and mole fractions */
void CKEQXR(double *  rho, double *  T, double *  x, double *  eqcon)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double gort[14]; /* temporary storage */

    /*compute the Gibbs free energy */
    gibbs(gort, tc);

    /*compute the equilibrium constants */
    equilibriumConstants(eqcon, gort, tT);

    /*reaction 1: O + HO2 => OH + O2 */
    /*eqcon[0] *= 1;  */

    /*reaction 2: H + HO2 => O + H2O */
    /*eqcon[1] *= 1;  */

    /*reaction 3: H + H2O2 <=> OH + H2O */
    /*eqcon[2] *= 1;  */

    /*reaction 4: O + CH => H + CO */
    /*eqcon[3] *= 1;  */

    /*reaction 5: H + CH <=> C + H2 */
    /*eqcon[4] *= 1;  */

    /*reaction 6: O + CH2 <=> H + HCO */
    /*eqcon[5] *= 1;  */

    /*reaction 7: H + O2 <=> O + OH */
    /*eqcon[6] *= 1;  */

    /*reaction 8: H + HO2 <=> 2.000000 OH */
    /*eqcon[7] *= 1;  */

    /*reaction 9: OH + CO <=> H + CO2 */
    /*eqcon[8] *= 1;  */

    /*reaction 10: OH + CH <=> H + HCO */
    /*eqcon[9] *= 1;  */
}

#ifdef AMREX_USE_CUDA
/*GPU version of productionRate: no more use of thermo namespace vectors */
/*compute the production rate for each species */
AMREX_GPU_HOST_DEVICE inline void  productionRate(double * wdot, double * sc, double T)
{
    double tc[] = { log(T), T, T*T, T*T*T, T*T*T*T }; /*temperature cache */
    double invT = 1.0 / tc[1];

    double qdot, q_f[10], q_r[10];
    comp_qfqr(q_f, q_r, sc, tc, invT);

    for (int i = 0; i < 14; ++i) {
        wdot[i] = 0.0;
    }

    qdot = q_f[0]-q_r[0];
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[4] += qdot;
    wdot[6] -= qdot;

    qdot = q_f[1]-q_r[1];
    wdot[1] -= qdot;
    wdot[2] += qdot;
    wdot[5] += qdot;
    wdot[6] -= qdot;

    qdot = q_f[2]-q_r[2];
    wdot[1] -= qdot;
    wdot[4] += qdot;
    wdot[5] += qdot;
    wdot[7] -= qdot;

    qdot = q_f[3]-q_r[3];
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[9] -= qdot;
    wdot[11] += qdot;

    qdot = q_f[4]-q_r[4];
    wdot[0] += qdot;
    wdot[1] -= qdot;
    wdot[8] += qdot;
    wdot[9] -= qdot;

    qdot = q_f[5]-q_r[5];
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[10] -= qdot;
    wdot[13] += qdot;

    qdot = q_f[6]-q_r[6];
    wdot[1] -= qdot;
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[4] += qdot;

    qdot = q_f[7]-q_r[7];
    wdot[1] -= qdot;
    wdot[4] += 2.000000 * qdot;
    wdot[6] -= qdot;

    qdot = q_f[8]-q_r[8];
    wdot[1] += qdot;
    wdot[4] -= qdot;
    wdot[11] -= qdot;
    wdot[12] += qdot;

    qdot = q_f[9]-q_r[9];
    wdot[1] += qdot;
    wdot[4] -= qdot;
    wdot[9] -= qdot;
    wdot[13] += qdot;

    return;
}

AMREX_GPU_HOST_DEVICE inline void comp_qfqr(double *  qf, double * qr, double * sc, double * tc, double invT)
{

    /*reaction 1: O + HO2 => OH + O2 */
    qf[0] = sc[2]*sc[6];
    qr[0] = 0.0;

    /*reaction 2: H + HO2 => O + H2O */
    qf[1] = sc[1]*sc[6];
    qr[1] = 0.0;

    /*reaction 3: H + H2O2 <=> OH + H2O */
    qf[2] = sc[1]*sc[7];
    qr[2] = sc[4]*sc[5];

    /*reaction 4: O + CH => H + CO */
    qf[3] = sc[2]*sc[9];
    qr[3] = 0.0;

    /*reaction 5: H + CH <=> C + H2 */
    qf[4] = sc[1]*sc[9];
    qr[4] = sc[0]*sc[8];

    /*reaction 6: O + CH2 <=> H + HCO */
    qf[5] = sc[2]*sc[10];
    qr[5] = sc[1]*sc[13];

    /*reaction 7: H + O2 <=> O + OH */
    qf[6] = sc[1]*sc[3];
    qr[6] = sc[2]*sc[4];

    /*reaction 8: H + HO2 <=> 2.000000 OH */
    qf[7] = sc[1]*sc[6];
    qr[7] = pow(sc[4], 2.000000);

    /*reaction 9: OH + CO <=> H + CO2 */
    qf[8] = sc[4]*sc[11];
    qr[8] = sc[1]*sc[12];

    /*reaction 10: OH + CH <=> H + HCO */
    qf[9] = sc[4]*sc[9];
    qr[9] = sc[1]*sc[13];

    /*compute the mixture concentration */
    double mixture = 0.0;
    for (int i = 0; i < 14; ++i) {
        mixture += sc[i];
    }

    /*compute the Gibbs free energy */
    double g_RT[14];
    gibbs(g_RT, tc);

    /*reference concentration: P_atm / (RT) in inverse mol/m^3 */
    double refC = 101325 / 8.31446 * invT;
    double refCinv = 1 / refC;

    /* Evaluate the kfs */
    double k_f, Corr;

    // (0):  O + HO2 => OH + O2
    k_f = 1.0000000000000002e-06 * 20000000000000 
               * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    Corr  = 1.0;
    qf[0] *= Corr * k_f;
    qr[0] *= Corr * k_f / exp(g_RT[2] - g_RT[3] - g_RT[4] + g_RT[6]);
    // (1):  H + HO2 => O + H2O
    k_f = 1.0000000000000002e-06 * 3970000000000 
               * exp(0 * tc[0] - 0.50321666580471969 * (671) * invT);
    Corr  = 1.0;
    qf[1] *= Corr * k_f;
    qr[1] *= Corr * k_f / exp(g_RT[1] - g_RT[2] - g_RT[5] + g_RT[6]);
    // (2):  H + H2O2 <=> OH + H2O
    k_f = 1.0000000000000002e-06 * 10000000000000 
               * exp(0 * tc[0] - 0.50321666580471969 * (3600) * invT);
    Corr  = 1.0;
    qf[2] *= Corr * k_f;
    qr[2] *= Corr * k_f / exp(g_RT[1] - g_RT[4] - g_RT[5] + g_RT[7]);
    // (3):  O + CH => H + CO
    k_f = 1.0000000000000002e-06 * 57000000000000 
               * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    Corr  = 1.0;
    qf[3] *= Corr * k_f;
    qr[3] *= Corr * k_f / exp(-g_RT[1] + g_RT[2] + g_RT[9] - g_RT[11]);
    // (4):  H + CH <=> C + H2
    k_f = 1.0000000000000002e-06 * 110000000000000 
               * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    Corr  = 1.0;
    qf[4] *= Corr * k_f;
    qr[4] *= Corr * k_f / exp(-g_RT[0] + g_RT[1] - g_RT[8] + g_RT[9]);
    // (5):  O + CH2 <=> H + HCO
    k_f = 1.0000000000000002e-06 * 80000000000000 
               * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    Corr  = 1.0;
    qf[5] *= Corr * k_f;
    qr[5] *= Corr * k_f / exp(-g_RT[1] + g_RT[2] + g_RT[10] - g_RT[13]);
    // (6):  H + O2 <=> O + OH
    k_f = 1.0000000000000002e-06 * 83000000000000 
               * exp(0 * tc[0] - 0.50321666580471969 * (14413) * invT);
    Corr  = 1.0;
    qf[6] *= Corr * k_f;
    qr[6] *= Corr * k_f / exp(g_RT[1] - g_RT[2] + g_RT[3] - g_RT[4]);
    // (7):  H + HO2 <=> 2.000000 OH
    k_f = 1.0000000000000002e-06 * 134000000000000 
               * exp(0 * tc[0] - 0.50321666580471969 * (635) * invT);
    Corr  = 1.0;
    qf[7] *= Corr * k_f;
    qr[7] *= Corr * k_f / exp(g_RT[1] - 2.000000*g_RT[4] + g_RT[6]);
    // (8):  OH + CO <=> H + CO2
    k_f = 1.0000000000000002e-06 * 47600000 
               * exp(1.228 * tc[0] - 0.50321666580471969 * (70) * invT);
    Corr  = 1.0;
    qf[8] *= Corr * k_f;
    qr[8] *= Corr * k_f / exp(-g_RT[1] + g_RT[4] + g_RT[11] - g_RT[12]);
    // (9):  OH + CH <=> H + HCO
    k_f = 1.0000000000000002e-06 * 30000000000000 
               * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    Corr  = 1.0;
    qf[9] *= Corr * k_f;
    qr[9] *= Corr * k_f / exp(-g_RT[1] + g_RT[4] + g_RT[9] - g_RT[13]);


    return;
}
#endif

/*Does this write to the file? */

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

    for (int i = 0; i < 14; ++i) {
        wdot[i] = 0.0;
    }

    qdot = q_f[0]-q_r[0];
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[4] += qdot;
    wdot[6] -= qdot;

    qdot = q_f[1]-q_r[1];
    wdot[1] -= qdot;
    wdot[2] += qdot;
    wdot[5] += qdot;
    wdot[6] -= qdot;

    qdot = q_f[2]-q_r[2];
    wdot[1] -= qdot;
    wdot[4] += qdot;
    wdot[5] += qdot;
    wdot[7] -= qdot;

    qdot = q_f[3]-q_r[3];
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[9] -= qdot;
    wdot[11] += qdot;

    qdot = q_f[4]-q_r[4];
    wdot[0] += qdot;
    wdot[1] -= qdot;
    wdot[8] += qdot;
    wdot[9] -= qdot;

    qdot = q_f[5]-q_r[5];
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[10] -= qdot;
    wdot[13] += qdot;

    qdot = q_f[6]-q_r[6];
    wdot[1] -= qdot;
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[4] += qdot;

    qdot = q_f[7]-q_r[7];
    wdot[1] -= qdot;
    wdot[4] += 2.000000 * qdot;
    wdot[6] -= qdot;

    qdot = q_f[8]-q_r[8];
    wdot[1] += qdot;
    wdot[4] -= qdot;
    wdot[11] -= qdot;
    wdot[12] += qdot;

    qdot = q_f[9]-q_r[9];
    wdot[1] += qdot;
    wdot[4] -= qdot;
    wdot[9] -= qdot;
    wdot[13] += qdot;

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
    double g_RT[14];
    gibbs(g_RT, tc);

    Kc[0] = g_RT[2] - g_RT[3] - g_RT[4] + g_RT[6];
    Kc[1] = g_RT[1] - g_RT[2] - g_RT[5] + g_RT[6];
    Kc[2] = g_RT[1] - g_RT[4] - g_RT[5] + g_RT[7];
    Kc[3] = -g_RT[1] + g_RT[2] + g_RT[9] - g_RT[11];
    Kc[4] = -g_RT[0] + g_RT[1] - g_RT[8] + g_RT[9];
    Kc[5] = -g_RT[1] + g_RT[2] + g_RT[10] - g_RT[13];
    Kc[6] = g_RT[1] - g_RT[2] + g_RT[3] - g_RT[4];
    Kc[7] = g_RT[1] - 2.000000*g_RT[4] + g_RT[6];
    Kc[8] = -g_RT[1] + g_RT[4] + g_RT[11] - g_RT[12];
    Kc[9] = -g_RT[1] + g_RT[4] + g_RT[9] - g_RT[13];

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
    qr[2] = sc[4]*qss_sc[1];

    /*reaction 4: O + CH => H + CO */
    qf[3] = sc[2]*qss_sc[5];
    qr[3] = 0.0;

    /*reaction 5: H + CH <=> C + H2 */
    qf[4] = sc[1]*qss_sc[5];
    qr[4] = sc[0]*qss_sc[4];

    /*reaction 6: O + CH2 <=> H + HCO */
    qf[5] = sc[2]*qss_sc[6];
    qr[5] = sc[1]*sc[13];

    /*reaction 7: H + O2 <=> O + OH */
    qf[6] = sc[1]*qss_sc[0];
    qr[6] = sc[2]*sc[4];

    /*reaction 8: H + HO2 <=> 2.000000 OH */
    qf[7] = sc[1]*qss_sc[2];
    qr[7] = pow(sc[4], 2.000000);

    /*reaction 9: OH + CO <=> H + CO2 */
    qf[8] = sc[4]*qss_sc[7];
    qr[8] = sc[1]*sc[12];

    /*reaction 10: OH + CH <=> H + HCO */
    qf[9] = sc[4]*qss_sc[5];
    qr[9] = sc[1]*sc[13];

    double T = tc[1];

    /*compute the mixture concentration */
    double mixture = 0.0;
    for (int i = 0; i < 14; ++i) {
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


#ifndef AMREX_USE_CUDA
/*compute the production rate for each species */
void vproductionRate(int npt, double *  wdot, double *  sc, double *  T)
{
    double k_f_s[10*npt], Kc_s[10*npt], mixture[npt], g_RT[14*npt];
    double tc[5*npt], invT[npt];

#ifdef __INTEL_COMPILER
     #pragma simd
#endif
    for (int i=0; i<npt; i++) {
        tc[0*npt+i] = log(T[i]);
        tc[1*npt+i] = T[i];
        tc[2*npt+i] = T[i]*T[i];
        tc[3*npt+i] = T[i]*T[i]*T[i];
        tc[4*npt+i] = T[i]*T[i]*T[i]*T[i];
        invT[i] = 1.0 / T[i];
    }

    for (int i=0; i<npt; i++) {
        mixture[i] = 0.0;
    }

    for (int n=0; n<14; n++) {
        for (int i=0; i<npt; i++) {
            mixture[i] += sc[n*npt+i];
            wdot[n*npt+i] = 0.0;
        }
    }

    vcomp_k_f(npt, k_f_s, tc, invT);

    vcomp_gibbs(npt, g_RT, tc);

    vcomp_Kc(npt, Kc_s, g_RT, invT);

    vcomp_wdot(npt, wdot, mixture, sc, k_f_s, Kc_s, tc, invT, T);
}

void vcomp_k_f(int npt, double *  k_f_s, double *  tc, double *  invT)
{
#ifdef __INTEL_COMPILER
    #pragma simd
#endif
    for (int i=0; i<npt; i++) {
        k_f_s[0*npt+i] = prefactor_units[0] * fwd_A[0] * exp(fwd_beta[0] * tc[i] - activation_units[0] * fwd_Ea[0] * invT[i]);
        k_f_s[1*npt+i] = prefactor_units[1] * fwd_A[1] * exp(fwd_beta[1] * tc[i] - activation_units[1] * fwd_Ea[1] * invT[i]);
        k_f_s[2*npt+i] = prefactor_units[2] * fwd_A[2] * exp(fwd_beta[2] * tc[i] - activation_units[2] * fwd_Ea[2] * invT[i]);
        k_f_s[3*npt+i] = prefactor_units[3] * fwd_A[3] * exp(fwd_beta[3] * tc[i] - activation_units[3] * fwd_Ea[3] * invT[i]);
        k_f_s[4*npt+i] = prefactor_units[4] * fwd_A[4] * exp(fwd_beta[4] * tc[i] - activation_units[4] * fwd_Ea[4] * invT[i]);
        k_f_s[5*npt+i] = prefactor_units[5] * fwd_A[5] * exp(fwd_beta[5] * tc[i] - activation_units[5] * fwd_Ea[5] * invT[i]);
        k_f_s[6*npt+i] = prefactor_units[6] * fwd_A[6] * exp(fwd_beta[6] * tc[i] - activation_units[6] * fwd_Ea[6] * invT[i]);
        k_f_s[7*npt+i] = prefactor_units[7] * fwd_A[7] * exp(fwd_beta[7] * tc[i] - activation_units[7] * fwd_Ea[7] * invT[i]);
        k_f_s[8*npt+i] = prefactor_units[8] * fwd_A[8] * exp(fwd_beta[8] * tc[i] - activation_units[8] * fwd_Ea[8] * invT[i]);
        k_f_s[9*npt+i] = prefactor_units[9] * fwd_A[9] * exp(fwd_beta[9] * tc[i] - activation_units[9] * fwd_Ea[9] * invT[i]);
    }
}

void vcomp_gibbs(int npt, double *  g_RT, double *  tc)
{
    /*compute the Gibbs free energy */
    for (int i=0; i<npt; i++) {
        double tg[5], g[14];
        tg[0] = tc[0*npt+i];
        tg[1] = tc[1*npt+i];
        tg[2] = tc[2*npt+i];
        tg[3] = tc[3*npt+i];
        tg[4] = tc[4*npt+i];

        gibbs(g, tg);

        g_RT[0*npt+i] = g[0];
        g_RT[1*npt+i] = g[1];
        g_RT[2*npt+i] = g[2];
        g_RT[3*npt+i] = g[3];
        g_RT[4*npt+i] = g[4];
        g_RT[5*npt+i] = g[5];
        g_RT[6*npt+i] = g[6];
        g_RT[7*npt+i] = g[7];
        g_RT[8*npt+i] = g[8];
        g_RT[9*npt+i] = g[9];
        g_RT[10*npt+i] = g[10];
        g_RT[11*npt+i] = g[11];
        g_RT[12*npt+i] = g[12];
        g_RT[13*npt+i] = g[13];
    }
}

void vcomp_Kc(int npt, double *  Kc_s, double *  g_RT, double *  invT)
{
#ifdef __INTEL_COMPILER
    #pragma simd
#endif
    for (int i=0; i<npt; i++) {
        /*reference concentration: P_atm / (RT) in inverse mol/m^3 */
        double refC = (101325. / 8.31451) * invT[i];
        double refCinv = 1.0 / refC;

        Kc_s[0*npt+i] = exp((g_RT[2*npt+i] + g_RT[6*npt+i]) - (g_RT[3*npt+i] + g_RT[4*npt+i]));
        Kc_s[1*npt+i] = exp((g_RT[1*npt+i] + g_RT[6*npt+i]) - (g_RT[2*npt+i] + g_RT[5*npt+i]));
        Kc_s[2*npt+i] = exp((g_RT[1*npt+i] + g_RT[7*npt+i]) - (g_RT[4*npt+i] + g_RT[5*npt+i]));
        Kc_s[3*npt+i] = exp((g_RT[2*npt+i] + g_RT[9*npt+i]) - (g_RT[1*npt+i] + g_RT[11*npt+i]));
        Kc_s[4*npt+i] = exp((g_RT[1*npt+i] + g_RT[9*npt+i]) - (g_RT[0*npt+i] + g_RT[8*npt+i]));
        Kc_s[5*npt+i] = exp((g_RT[2*npt+i] + g_RT[10*npt+i]) - (g_RT[1*npt+i] + g_RT[13*npt+i]));
        Kc_s[6*npt+i] = exp((g_RT[1*npt+i] + g_RT[3*npt+i]) - (g_RT[2*npt+i] + g_RT[4*npt+i]));
        Kc_s[7*npt+i] = exp((g_RT[1*npt+i] + g_RT[6*npt+i]) - (2.000000 * g_RT[4*npt+i]));
        Kc_s[8*npt+i] = exp((g_RT[4*npt+i] + g_RT[11*npt+i]) - (g_RT[1*npt+i] + g_RT[12*npt+i]));
        Kc_s[9*npt+i] = exp((g_RT[4*npt+i] + g_RT[9*npt+i]) - (g_RT[1*npt+i] + g_RT[13*npt+i]));
    }
}

void vcomp_wdot(int npt, double *  wdot, double *  mixture, double *  sc,
		double *  k_f_s, double *  Kc_s,
		double *  tc, double *  invT, double *  T)
{
#ifdef __INTEL_COMPILER
    #pragma simd
#endif
    for (int i=0; i<npt; i++) {
        double qdot, q_f, q_r, phi_f, phi_r, k_f, k_r, Kc;
        double alpha;

        /*reaction 1: O + HO2 => OH + O2 */
        phi_f = sc[2*npt+i]*sc[6*npt+i];
        k_f = k_f_s[0*npt+i];
        q_f = phi_f * k_f;
        q_r = 0.0;
        qdot = q_f - q_r;
        wdot[2*npt+i] -= qdot;
        wdot[3*npt+i] += qdot;
        wdot[4*npt+i] += qdot;
        wdot[6*npt+i] -= qdot;

        /*reaction 2: H + HO2 => O + H2O */
        phi_f = sc[1*npt+i]*sc[6*npt+i];
        k_f = k_f_s[1*npt+i];
        q_f = phi_f * k_f;
        q_r = 0.0;
        qdot = q_f - q_r;
        wdot[1*npt+i] -= qdot;
        wdot[2*npt+i] += qdot;
        wdot[5*npt+i] += qdot;
        wdot[6*npt+i] -= qdot;

        /*reaction 3: H + H2O2 <=> OH + H2O */
        phi_f = sc[1*npt+i]*sc[7*npt+i];
        k_f = k_f_s[2*npt+i];
        q_f = phi_f * k_f;
        phi_r = sc[4*npt+i]*sc[5*npt+i];
        Kc = Kc_s[2*npt+i];
        k_r = k_f / Kc;
        q_r = phi_r * k_r;
        qdot = q_f - q_r;
        wdot[1*npt+i] -= qdot;
        wdot[4*npt+i] += qdot;
        wdot[5*npt+i] += qdot;
        wdot[7*npt+i] -= qdot;

        /*reaction 4: O + CH => H + CO */
        phi_f = sc[2*npt+i]*sc[9*npt+i];
        k_f = k_f_s[3*npt+i];
        q_f = phi_f * k_f;
        q_r = 0.0;
        qdot = q_f - q_r;
        wdot[1*npt+i] += qdot;
        wdot[2*npt+i] -= qdot;
        wdot[9*npt+i] -= qdot;
        wdot[11*npt+i] += qdot;

        /*reaction 5: H + CH <=> C + H2 */
        phi_f = sc[1*npt+i]*sc[9*npt+i];
        k_f = k_f_s[4*npt+i];
        q_f = phi_f * k_f;
        phi_r = sc[0*npt+i]*sc[8*npt+i];
        Kc = Kc_s[4*npt+i];
        k_r = k_f / Kc;
        q_r = phi_r * k_r;
        qdot = q_f - q_r;
        wdot[0*npt+i] += qdot;
        wdot[1*npt+i] -= qdot;
        wdot[8*npt+i] += qdot;
        wdot[9*npt+i] -= qdot;

        /*reaction 6: O + CH2 <=> H + HCO */
        phi_f = sc[2*npt+i]*sc[10*npt+i];
        k_f = k_f_s[5*npt+i];
        q_f = phi_f * k_f;
        phi_r = sc[1*npt+i]*sc[13*npt+i];
        Kc = Kc_s[5*npt+i];
        k_r = k_f / Kc;
        q_r = phi_r * k_r;
        qdot = q_f - q_r;
        wdot[1*npt+i] += qdot;
        wdot[2*npt+i] -= qdot;
        wdot[10*npt+i] -= qdot;
        wdot[13*npt+i] += qdot;

        /*reaction 7: H + O2 <=> O + OH */
        phi_f = sc[1*npt+i]*sc[3*npt+i];
        k_f = k_f_s[6*npt+i];
        q_f = phi_f * k_f;
        phi_r = sc[2*npt+i]*sc[4*npt+i];
        Kc = Kc_s[6*npt+i];
        k_r = k_f / Kc;
        q_r = phi_r * k_r;
        qdot = q_f - q_r;
        wdot[1*npt+i] -= qdot;
        wdot[2*npt+i] += qdot;
        wdot[3*npt+i] -= qdot;
        wdot[4*npt+i] += qdot;

        /*reaction 8: H + HO2 <=> 2.000000 OH */
        phi_f = sc[1*npt+i]*sc[6*npt+i];
        k_f = k_f_s[7*npt+i];
        q_f = phi_f * k_f;
        phi_r = pow(sc[4*npt+i], 2.000000);
        Kc = Kc_s[7*npt+i];
        k_r = k_f / Kc;
        q_r = phi_r * k_r;
        qdot = q_f - q_r;
        wdot[1*npt+i] -= qdot;
        wdot[4*npt+i] += 2.000000 * qdot;
        wdot[6*npt+i] -= qdot;

        /*reaction 9: OH + CO <=> H + CO2 */
        phi_f = sc[4*npt+i]*sc[11*npt+i];
        k_f = k_f_s[8*npt+i];
        q_f = phi_f * k_f;
        phi_r = sc[1*npt+i]*sc[12*npt+i];
        Kc = Kc_s[8*npt+i];
        k_r = k_f / Kc;
        q_r = phi_r * k_r;
        qdot = q_f - q_r;
        wdot[1*npt+i] += qdot;
        wdot[4*npt+i] -= qdot;
        wdot[11*npt+i] -= qdot;
        wdot[12*npt+i] += qdot;

        /*reaction 10: OH + CH <=> H + HCO */
        phi_f = sc[4*npt+i]*sc[9*npt+i];
        k_f = k_f_s[9*npt+i];
        q_f = phi_f * k_f;
        phi_r = sc[1*npt+i]*sc[13*npt+i];
        Kc = Kc_s[9*npt+i];
        k_r = k_f / Kc;
        q_r = phi_r * k_r;
        qdot = q_f - q_r;
        wdot[1*npt+i] += qdot;
        wdot[4*npt+i] -= qdot;
        wdot[9*npt+i] -= qdot;
        wdot[13*npt+i] += qdot;
    }
}
#endif

/*compute an approx to the reaction Jacobian (for preconditioning) */
AMREX_GPU_HOST_DEVICE void DWDOT_SIMPLIFIED(double *  J, double *  sc, double *  Tp, int * HP)
{
    double c[14];

    for (int k=0; k<14; k++) {
        c[k] = 1.e6 * sc[k];
    }

    aJacobian_precond(J, c, *Tp, *HP);

    /* dwdot[k]/dT */
    /* dTdot/d[X] */
    for (int k=0; k<14; k++) {
        J[210+k] *= 1.e-6;
        J[k*15+14] *= 1.e6;
    }

    return;
}

/*compute the reaction Jacobian */
AMREX_GPU_HOST_DEVICE void DWDOT(double *  J, double *  sc, double *  Tp, int * consP)
{
    double c[14];

    for (int k=0; k<14; k++) {
        c[k] = 1.e6 * sc[k];
    }

    aJacobian(J, c, *Tp, *consP);

    /* dwdot[k]/dT */
    /* dTdot/d[X] */
    for (int k=0; k<14; k++) {
        J[210+k] *= 1.e-6;
        J[k*15+14] *= 1.e6;
    }

    return;
}

/*compute the sparsity pattern of the chemistry Jacobian */
AMREX_GPU_HOST_DEVICE void SPARSITY_INFO( int * nJdata, int * consP, int NCELLS)
{
    double c[14];
    double J[225];

    for (int k=0; k<14; k++) {
        c[k] = 1.0/ 14.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    int nJdata_tmp = 0;
    for (int k=0; k<15; k++) {
        for (int l=0; l<15; l++) {
            if(J[ 15 * k + l] != 0.0){
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
    double c[14];
    double J[225];

    for (int k=0; k<14; k++) {
        c[k] = 1.0/ 14.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    int nJdata_tmp = 0;
    for (int k=0; k<15; k++) {
        for (int l=0; l<15; l++) {
            if(k == l){
                nJdata_tmp = nJdata_tmp + 1;
            } else {
                if(J[ 15 * k + l] != 0.0){
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
    double c[14];
    double J[225];

    for (int k=0; k<14; k++) {
        c[k] = 1.0/ 14.000000 ;
    }

    aJacobian_precond(J, c, 1500.0, *consP);

    int nJdata_tmp = 0;
    for (int k=0; k<15; k++) {
        for (int l=0; l<15; l++) {
            if(k == l){
                nJdata_tmp = nJdata_tmp + 1;
            } else {
                if(J[ 15 * k + l] != 0.0){
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
    double c[14];
    double J[225];
    int offset_row;
    int offset_col;

    for (int k=0; k<14; k++) {
        c[k] = 1.0/ 14.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    colPtrs[0] = 0;
    int nJdata_tmp = 0;
    for (int nc=0; nc<NCELLS; nc++) {
        offset_row = nc * 15;
        offset_col = nc * 15;
        for (int k=0; k<15; k++) {
            for (int l=0; l<15; l++) {
                if(J[15*k + l] != 0.0) {
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
    double c[14];
    double J[225];
    int offset;

    for (int k=0; k<14; k++) {
        c[k] = 1.0/ 14.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    if (base == 1) {
        rowPtrs[0] = 1;
        int nJdata_tmp = 1;
        for (int nc=0; nc<NCELLS; nc++) {
            offset = nc * 15;
            for (int l=0; l<15; l++) {
                for (int k=0; k<15; k++) {
                    if(J[15*k + l] != 0.0) {
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
            offset = nc * 15;
            for (int l=0; l<15; l++) {
                for (int k=0; k<15; k++) {
                    if(J[15*k + l] != 0.0) {
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
    double c[14];
    double J[225];
    int offset;

    for (int k=0; k<14; k++) {
        c[k] = 1.0/ 14.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    if (base == 1) {
        rowPtr[0] = 1;
        int nJdata_tmp = 1;
        for (int nc=0; nc<NCELLS; nc++) {
            offset = nc * 15;
            for (int l=0; l<15; l++) {
                for (int k=0; k<15; k++) {
                    if (k == l) {
                        colVals[nJdata_tmp-1] = l+1 + offset; 
                        nJdata_tmp = nJdata_tmp + 1; 
                    } else {
                        if(J[15*k + l] != 0.0) {
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
            offset = nc * 15;
            for (int l=0; l<15; l++) {
                for (int k=0; k<15; k++) {
                    if (k == l) {
                        colVals[nJdata_tmp] = l + offset; 
                        nJdata_tmp = nJdata_tmp + 1; 
                    } else {
                        if(J[15*k + l] != 0.0) {
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
    double c[14];
    double J[225];

    for (int k=0; k<14; k++) {
        c[k] = 1.0/ 14.000000 ;
    }

    aJacobian_precond(J, c, 1500.0, *consP);

    colPtrs[0] = 0;
    int nJdata_tmp = 0;
    for (int k=0; k<15; k++) {
        for (int l=0; l<15; l++) {
            if (k == l) {
                rowVals[nJdata_tmp] = l; 
                indx[nJdata_tmp] = 15*k + l;
                nJdata_tmp = nJdata_tmp + 1; 
            } else {
                if(J[15*k + l] != 0.0) {
                    rowVals[nJdata_tmp] = l; 
                    indx[nJdata_tmp] = 15*k + l;
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
    double c[14];
    double J[225];

    for (int k=0; k<14; k++) {
        c[k] = 1.0/ 14.000000 ;
    }

    aJacobian_precond(J, c, 1500.0, *consP);

    if (base == 1) {
        rowPtr[0] = 1;
        int nJdata_tmp = 1;
        for (int l=0; l<15; l++) {
            for (int k=0; k<15; k++) {
                if (k == l) {
                    colVals[nJdata_tmp-1] = l+1; 
                    nJdata_tmp = nJdata_tmp + 1; 
                } else {
                    if(J[15*k + l] != 0.0) {
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
        for (int l=0; l<15; l++) {
            for (int k=0; k<15; k++) {
                if (k == l) {
                    colVals[nJdata_tmp] = l; 
                    nJdata_tmp = nJdata_tmp + 1; 
                } else {
                    if(J[15*k + l] != 0.0) {
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


#ifdef AMREX_USE_CUDA
/*compute the reaction Jacobian on GPU */
AMREX_GPU_HOST_DEVICE
void aJacobian(double * J, double * sc, double T, int consP)
{


    for (int i=0; i<225; i++) {
        J[i] = 0.0;
    }

    double wdot[14];
    for (int k=0; k<14; k++) {
        wdot[k] = 0.0;
    }

    double tc[] = { log(T), T, T*T, T*T*T, T*T*T*T }; /*temperature cache */
    double invT = 1.0 / tc[1];
    double invT2 = invT * invT;

    /*reference concentration: P_atm / (RT) in inverse mol/m^3 */
    double refC = 101325 / 8.31446 / T;
    double refCinv = 1.0 / refC;

    /*compute the mixture concentration */
    double mixture = 0.0;
    for (int k = 0; k < 14; ++k) {
        mixture += sc[k];
    }

    /*compute the Gibbs free energy */
    double g_RT[14];
    gibbs(g_RT, tc);

    /*compute the species enthalpy */
    double h_RT[14];
    speciesEnthalpy(h_RT, tc);

    double phi_f, k_f, k_r, phi_r, Kc, q, q_nocor, Corr, alpha;
    double dlnkfdT, dlnk0dT, dlnKcdT, dkrdT, dqdT;
    double dqdci, dcdc_fac, dqdc[14];
    double Pr, fPr, F, k_0, logPr;
    double logFcent, troe_c, troe_n, troePr_den, troePr, troe;
    double Fcent1, Fcent2, Fcent3, Fcent;
    double dlogFdc, dlogFdn, dlogFdcn_fac;
    double dlogPrdT, dlogfPrdT, dlogFdT, dlogFcentdT, dlogFdlogPr, dlnCorrdT;
    const double ln10 = log(10.0);
    const double log10e = 1.0/log(10.0);
    /*reaction 1: O + HO2 => OH + O2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[2]*sc[6];
    k_f = 1.0000000000000002e-06 * 20000000000000
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  0  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[2] -= q; /* O */
    wdot[3] += q; /* O2 */
    wdot[4] += q; /* OH */
    wdot[6] -= q; /* HO2 */
    /* d()/d[O] */
    dqdci =  + k_f*sc[6];
    J[32] -= dqdci;               /* dwdot[O]/d[O] */
    J[33] += dqdci;               /* dwdot[O2]/d[O] */
    J[34] += dqdci;               /* dwdot[OH]/d[O] */
    J[36] -= dqdci;               /* dwdot[HO2]/d[O] */
    /* d()/d[HO2] */
    dqdci =  + k_f*sc[2];
    J[92] -= dqdci;               /* dwdot[O]/d[HO2] */
    J[93] += dqdci;               /* dwdot[O2]/d[HO2] */
    J[94] += dqdci;               /* dwdot[OH]/d[HO2] */
    J[96] -= dqdci;               /* dwdot[HO2]/d[HO2] */
    /* d()/dT */
    J[212] -= dqdT;               /* dwdot[O]/dT */
    J[213] += dqdT;               /* dwdot[O2]/dT */
    J[214] += dqdT;               /* dwdot[OH]/dT */
    J[216] -= dqdT;               /* dwdot[HO2]/dT */

    /*reaction 2: H + HO2 => O + H2O */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[6];
    k_f = 1.0000000000000002e-06 * 3970000000000
                * exp(0 * tc[0] - 0.50321666580471969 * (671) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  671  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[1] -= q; /* H */
    wdot[2] += q; /* O */
    wdot[5] += q; /* H2O */
    wdot[6] -= q; /* HO2 */
    /* d()/d[H] */
    dqdci =  + k_f*sc[6];
    J[16] -= dqdci;               /* dwdot[H]/d[H] */
    J[17] += dqdci;               /* dwdot[O]/d[H] */
    J[20] += dqdci;               /* dwdot[H2O]/d[H] */
    J[21] -= dqdci;               /* dwdot[HO2]/d[H] */
    /* d()/d[HO2] */
    dqdci =  + k_f*sc[1];
    J[91] -= dqdci;               /* dwdot[H]/d[HO2] */
    J[92] += dqdci;               /* dwdot[O]/d[HO2] */
    J[95] += dqdci;               /* dwdot[H2O]/d[HO2] */
    J[96] -= dqdci;               /* dwdot[HO2]/d[HO2] */
    /* d()/dT */
    J[211] -= dqdT;               /* dwdot[H]/dT */
    J[212] += dqdT;               /* dwdot[O]/dT */
    J[215] += dqdT;               /* dwdot[H2O]/dT */
    J[216] -= dqdT;               /* dwdot[HO2]/dT */

    /*reaction 3: H + H2O2 <=> OH + H2O */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[7];
    k_f = 1.0000000000000002e-06 * 10000000000000
                * exp(0 * tc[0] - 0.50321666580471969 * (3600) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  3600  * invT2;
    /* reverse */
    phi_r = sc[4]*sc[5];
    Kc = exp(g_RT[1] - g_RT[4] - g_RT[5] + g_RT[7]);
    k_r = k_f / Kc;
    dlnKcdT = invT * (-(h_RT[1] + h_RT[7]) + (h_RT[4] + h_RT[5]));
    dkrdT = (dlnkfdT - dlnKcdT)*k_r;
    /* rate of progress */
    q = k_f*phi_f - k_r*phi_r;
    dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
    /* update wdot */
    wdot[1] -= q; /* H */
    wdot[4] += q; /* OH */
    wdot[5] += q; /* H2O */
    wdot[7] -= q; /* H2O2 */
    /* d()/d[H] */
    dqdci =  + k_f*sc[7];
    J[16] -= dqdci;               /* dwdot[H]/d[H] */
    J[19] += dqdci;               /* dwdot[OH]/d[H] */
    J[20] += dqdci;               /* dwdot[H2O]/d[H] */
    J[22] -= dqdci;               /* dwdot[H2O2]/d[H] */
    /* d()/d[OH] */
    dqdci =  - k_r*sc[5];
    J[61] -= dqdci;               /* dwdot[H]/d[OH] */
    J[64] += dqdci;               /* dwdot[OH]/d[OH] */
    J[65] += dqdci;               /* dwdot[H2O]/d[OH] */
    J[67] -= dqdci;               /* dwdot[H2O2]/d[OH] */
    /* d()/d[H2O] */
    dqdci =  - k_r*sc[4];
    J[76] -= dqdci;               /* dwdot[H]/d[H2O] */
    J[79] += dqdci;               /* dwdot[OH]/d[H2O] */
    J[80] += dqdci;               /* dwdot[H2O]/d[H2O] */
    J[82] -= dqdci;               /* dwdot[H2O2]/d[H2O] */
    /* d()/d[H2O2] */
    dqdci =  + k_f*sc[1];
    J[106] -= dqdci;              /* dwdot[H]/d[H2O2] */
    J[109] += dqdci;              /* dwdot[OH]/d[H2O2] */
    J[110] += dqdci;              /* dwdot[H2O]/d[H2O2] */
    J[112] -= dqdci;              /* dwdot[H2O2]/d[H2O2] */
    /* d()/dT */
    J[211] -= dqdT;               /* dwdot[H]/dT */
    J[214] += dqdT;               /* dwdot[OH]/dT */
    J[215] += dqdT;               /* dwdot[H2O]/dT */
    J[217] -= dqdT;               /* dwdot[H2O2]/dT */

    /*reaction 4: O + CH => H + CO */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[2]*sc[9];
    k_f = 1.0000000000000002e-06 * 57000000000000
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  0  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[1] += q; /* H */
    wdot[2] -= q; /* O */
    wdot[9] -= q; /* CH */
    wdot[11] += q; /* CO */
    /* d()/d[O] */
    dqdci =  + k_f*sc[9];
    J[31] += dqdci;               /* dwdot[H]/d[O] */
    J[32] -= dqdci;               /* dwdot[O]/d[O] */
    J[39] -= dqdci;               /* dwdot[CH]/d[O] */
    J[41] += dqdci;               /* dwdot[CO]/d[O] */
    /* d()/d[CH] */
    dqdci =  + k_f*sc[2];
    J[136] += dqdci;              /* dwdot[H]/d[CH] */
    J[137] -= dqdci;              /* dwdot[O]/d[CH] */
    J[144] -= dqdci;              /* dwdot[CH]/d[CH] */
    J[146] += dqdci;              /* dwdot[CO]/d[CH] */
    /* d()/dT */
    J[211] += dqdT;               /* dwdot[H]/dT */
    J[212] -= dqdT;               /* dwdot[O]/dT */
    J[219] -= dqdT;               /* dwdot[CH]/dT */
    J[221] += dqdT;               /* dwdot[CO]/dT */

    /*reaction 5: H + CH <=> C + H2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[9];
    k_f = 1.0000000000000002e-06 * 110000000000000
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  0  * invT2;
    /* reverse */
    phi_r = sc[0]*sc[8];
    Kc = exp(-g_RT[0] + g_RT[1] - g_RT[8] + g_RT[9]);
    k_r = k_f / Kc;
    dlnKcdT = invT * (-(h_RT[1] + h_RT[9]) + (h_RT[0] + h_RT[8]));
    dkrdT = (dlnkfdT - dlnKcdT)*k_r;
    /* rate of progress */
    q = k_f*phi_f - k_r*phi_r;
    dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
    /* update wdot */
    wdot[0] += q; /* H2 */
    wdot[1] -= q; /* H */
    wdot[8] += q; /* C */
    wdot[9] -= q; /* CH */
    /* d()/d[H2] */
    dqdci =  - k_r*sc[8];
    J[0] += dqdci;                /* dwdot[H2]/d[H2] */
    J[1] -= dqdci;                /* dwdot[H]/d[H2] */
    J[8] += dqdci;                /* dwdot[C]/d[H2] */
    J[9] -= dqdci;                /* dwdot[CH]/d[H2] */
    /* d()/d[H] */
    dqdci =  + k_f*sc[9];
    J[15] += dqdci;               /* dwdot[H2]/d[H] */
    J[16] -= dqdci;               /* dwdot[H]/d[H] */
    J[23] += dqdci;               /* dwdot[C]/d[H] */
    J[24] -= dqdci;               /* dwdot[CH]/d[H] */
    /* d()/d[C] */
    dqdci =  - k_r*sc[0];
    J[120] += dqdci;              /* dwdot[H2]/d[C] */
    J[121] -= dqdci;              /* dwdot[H]/d[C] */
    J[128] += dqdci;              /* dwdot[C]/d[C] */
    J[129] -= dqdci;              /* dwdot[CH]/d[C] */
    /* d()/d[CH] */
    dqdci =  + k_f*sc[1];
    J[135] += dqdci;              /* dwdot[H2]/d[CH] */
    J[136] -= dqdci;              /* dwdot[H]/d[CH] */
    J[143] += dqdci;              /* dwdot[C]/d[CH] */
    J[144] -= dqdci;              /* dwdot[CH]/d[CH] */
    /* d()/dT */
    J[210] += dqdT;               /* dwdot[H2]/dT */
    J[211] -= dqdT;               /* dwdot[H]/dT */
    J[218] += dqdT;               /* dwdot[C]/dT */
    J[219] -= dqdT;               /* dwdot[CH]/dT */

    /*reaction 6: O + CH2 <=> H + HCO */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[2]*sc[10];
    k_f = 1.0000000000000002e-06 * 80000000000000
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  0  * invT2;
    /* reverse */
    phi_r = sc[1]*sc[13];
    Kc = exp(-g_RT[1] + g_RT[2] + g_RT[10] - g_RT[13]);
    k_r = k_f / Kc;
    dlnKcdT = invT * (-(h_RT[2] + h_RT[10]) + (h_RT[1] + h_RT[13]));
    dkrdT = (dlnkfdT - dlnKcdT)*k_r;
    /* rate of progress */
    q = k_f*phi_f - k_r*phi_r;
    dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
    /* update wdot */
    wdot[1] += q; /* H */
    wdot[2] -= q; /* O */
    wdot[10] -= q; /* CH2 */
    wdot[13] += q; /* HCO */
    /* d()/d[H] */
    dqdci =  - k_r*sc[13];
    J[16] += dqdci;               /* dwdot[H]/d[H] */
    J[17] -= dqdci;               /* dwdot[O]/d[H] */
    J[25] -= dqdci;               /* dwdot[CH2]/d[H] */
    J[28] += dqdci;               /* dwdot[HCO]/d[H] */
    /* d()/d[O] */
    dqdci =  + k_f*sc[10];
    J[31] += dqdci;               /* dwdot[H]/d[O] */
    J[32] -= dqdci;               /* dwdot[O]/d[O] */
    J[40] -= dqdci;               /* dwdot[CH2]/d[O] */
    J[43] += dqdci;               /* dwdot[HCO]/d[O] */
    /* d()/d[CH2] */
    dqdci =  + k_f*sc[2];
    J[151] += dqdci;              /* dwdot[H]/d[CH2] */
    J[152] -= dqdci;              /* dwdot[O]/d[CH2] */
    J[160] -= dqdci;              /* dwdot[CH2]/d[CH2] */
    J[163] += dqdci;              /* dwdot[HCO]/d[CH2] */
    /* d()/d[HCO] */
    dqdci =  - k_r*sc[1];
    J[196] += dqdci;              /* dwdot[H]/d[HCO] */
    J[197] -= dqdci;              /* dwdot[O]/d[HCO] */
    J[205] -= dqdci;              /* dwdot[CH2]/d[HCO] */
    J[208] += dqdci;              /* dwdot[HCO]/d[HCO] */
    /* d()/dT */
    J[211] += dqdT;               /* dwdot[H]/dT */
    J[212] -= dqdT;               /* dwdot[O]/dT */
    J[220] -= dqdT;               /* dwdot[CH2]/dT */
    J[223] += dqdT;               /* dwdot[HCO]/dT */

    /*reaction 7: H + O2 <=> O + OH */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[3];
    k_f = 1.0000000000000002e-06 * 83000000000000
                * exp(0 * tc[0] - 0.50321666580471969 * (14413) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  14413  * invT2;
    /* reverse */
    phi_r = sc[2]*sc[4];
    Kc = exp(g_RT[1] - g_RT[2] + g_RT[3] - g_RT[4]);
    k_r = k_f / Kc;
    dlnKcdT = invT * (-(h_RT[1] + h_RT[3]) + (h_RT[2] + h_RT[4]));
    dkrdT = (dlnkfdT - dlnKcdT)*k_r;
    /* rate of progress */
    q = k_f*phi_f - k_r*phi_r;
    dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
    /* update wdot */
    wdot[1] -= q; /* H */
    wdot[2] += q; /* O */
    wdot[3] -= q; /* O2 */
    wdot[4] += q; /* OH */
    /* d()/d[H] */
    dqdci =  + k_f*sc[3];
    J[16] -= dqdci;               /* dwdot[H]/d[H] */
    J[17] += dqdci;               /* dwdot[O]/d[H] */
    J[18] -= dqdci;               /* dwdot[O2]/d[H] */
    J[19] += dqdci;               /* dwdot[OH]/d[H] */
    /* d()/d[O] */
    dqdci =  - k_r*sc[4];
    J[31] -= dqdci;               /* dwdot[H]/d[O] */
    J[32] += dqdci;               /* dwdot[O]/d[O] */
    J[33] -= dqdci;               /* dwdot[O2]/d[O] */
    J[34] += dqdci;               /* dwdot[OH]/d[O] */
    /* d()/d[O2] */
    dqdci =  + k_f*sc[1];
    J[46] -= dqdci;               /* dwdot[H]/d[O2] */
    J[47] += dqdci;               /* dwdot[O]/d[O2] */
    J[48] -= dqdci;               /* dwdot[O2]/d[O2] */
    J[49] += dqdci;               /* dwdot[OH]/d[O2] */
    /* d()/d[OH] */
    dqdci =  - k_r*sc[2];
    J[61] -= dqdci;               /* dwdot[H]/d[OH] */
    J[62] += dqdci;               /* dwdot[O]/d[OH] */
    J[63] -= dqdci;               /* dwdot[O2]/d[OH] */
    J[64] += dqdci;               /* dwdot[OH]/d[OH] */
    /* d()/dT */
    J[211] -= dqdT;               /* dwdot[H]/dT */
    J[212] += dqdT;               /* dwdot[O]/dT */
    J[213] -= dqdT;               /* dwdot[O2]/dT */
    J[214] += dqdT;               /* dwdot[OH]/dT */

    /*reaction 8: H + HO2 <=> 2.000000 OH */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[6];
    k_f = 1.0000000000000002e-06 * 134000000000000
                * exp(0 * tc[0] - 0.50321666580471969 * (635) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  635  * invT2;
    /* reverse */
    phi_r = pow(sc[4], 2.000000);
    Kc = exp(g_RT[1] - 2.000000*g_RT[4] + g_RT[6]);
    k_r = k_f / Kc;
    dlnKcdT = invT * (-(h_RT[1] + h_RT[6]) + (2.000000*h_RT[4]));
    dkrdT = (dlnkfdT - dlnKcdT)*k_r;
    /* rate of progress */
    q = k_f*phi_f - k_r*phi_r;
    dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
    /* update wdot */
    wdot[1] -= q; /* H */
    wdot[4] += 2 * q; /* OH */
    wdot[6] -= q; /* HO2 */
    /* d()/d[H] */
    dqdci =  + k_f*sc[6];
    J[16] -= dqdci;               /* dwdot[H]/d[H] */
    J[19] += 2 * dqdci;           /* dwdot[OH]/d[H] */
    J[21] -= dqdci;               /* dwdot[HO2]/d[H] */
    /* d()/d[OH] */
    dqdci =  - k_r*2.000000*sc[4];
    J[61] -= dqdci;               /* dwdot[H]/d[OH] */
    J[64] += 2 * dqdci;           /* dwdot[OH]/d[OH] */
    J[66] -= dqdci;               /* dwdot[HO2]/d[OH] */
    /* d()/d[HO2] */
    dqdci =  + k_f*sc[1];
    J[91] -= dqdci;               /* dwdot[H]/d[HO2] */
    J[94] += 2 * dqdci;           /* dwdot[OH]/d[HO2] */
    J[96] -= dqdci;               /* dwdot[HO2]/d[HO2] */
    /* d()/dT */
    J[211] -= dqdT;               /* dwdot[H]/dT */
    J[214] += 2 * dqdT;           /* dwdot[OH]/dT */
    J[216] -= dqdT;               /* dwdot[HO2]/dT */

    /*reaction 9: OH + CO <=> H + CO2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[4]*sc[11];
    k_f = 1.0000000000000002e-06 * 47600000
                * exp(1.228 * tc[0] - 0.50321666580471969 * (70) * invT);
    dlnkfdT = 1.228 * invT + 0.50321666580471969 *  70  * invT2;
    /* reverse */
    phi_r = sc[1]*sc[12];
    Kc = exp(-g_RT[1] + g_RT[4] + g_RT[11] - g_RT[12]);
    k_r = k_f / Kc;
    dlnKcdT = invT * (-(h_RT[4] + h_RT[11]) + (h_RT[1] + h_RT[12]));
    dkrdT = (dlnkfdT - dlnKcdT)*k_r;
    /* rate of progress */
    q = k_f*phi_f - k_r*phi_r;
    dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
    /* update wdot */
    wdot[1] += q; /* H */
    wdot[4] -= q; /* OH */
    wdot[11] -= q; /* CO */
    wdot[12] += q; /* CO2 */
    /* d()/d[H] */
    dqdci =  - k_r*sc[12];
    J[16] += dqdci;               /* dwdot[H]/d[H] */
    J[19] -= dqdci;               /* dwdot[OH]/d[H] */
    J[26] -= dqdci;               /* dwdot[CO]/d[H] */
    J[27] += dqdci;               /* dwdot[CO2]/d[H] */
    /* d()/d[OH] */
    dqdci =  + k_f*sc[11];
    J[61] += dqdci;               /* dwdot[H]/d[OH] */
    J[64] -= dqdci;               /* dwdot[OH]/d[OH] */
    J[71] -= dqdci;               /* dwdot[CO]/d[OH] */
    J[72] += dqdci;               /* dwdot[CO2]/d[OH] */
    /* d()/d[CO] */
    dqdci =  + k_f*sc[4];
    J[166] += dqdci;              /* dwdot[H]/d[CO] */
    J[169] -= dqdci;              /* dwdot[OH]/d[CO] */
    J[176] -= dqdci;              /* dwdot[CO]/d[CO] */
    J[177] += dqdci;              /* dwdot[CO2]/d[CO] */
    /* d()/d[CO2] */
    dqdci =  - k_r*sc[1];
    J[181] += dqdci;              /* dwdot[H]/d[CO2] */
    J[184] -= dqdci;              /* dwdot[OH]/d[CO2] */
    J[191] -= dqdci;              /* dwdot[CO]/d[CO2] */
    J[192] += dqdci;              /* dwdot[CO2]/d[CO2] */
    /* d()/dT */
    J[211] += dqdT;               /* dwdot[H]/dT */
    J[214] -= dqdT;               /* dwdot[OH]/dT */
    J[221] -= dqdT;               /* dwdot[CO]/dT */
    J[222] += dqdT;               /* dwdot[CO2]/dT */

    /*reaction 10: OH + CH <=> H + HCO */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[4]*sc[9];
    k_f = 1.0000000000000002e-06 * 30000000000000
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  0  * invT2;
    /* reverse */
    phi_r = sc[1]*sc[13];
    Kc = exp(-g_RT[1] + g_RT[4] + g_RT[9] - g_RT[13]);
    k_r = k_f / Kc;
    dlnKcdT = invT * (-(h_RT[4] + h_RT[9]) + (h_RT[1] + h_RT[13]));
    dkrdT = (dlnkfdT - dlnKcdT)*k_r;
    /* rate of progress */
    q = k_f*phi_f - k_r*phi_r;
    dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
    /* update wdot */
    wdot[1] += q; /* H */
    wdot[4] -= q; /* OH */
    wdot[9] -= q; /* CH */
    wdot[13] += q; /* HCO */
    /* d()/d[H] */
    dqdci =  - k_r*sc[13];
    J[16] += dqdci;               /* dwdot[H]/d[H] */
    J[19] -= dqdci;               /* dwdot[OH]/d[H] */
    J[24] -= dqdci;               /* dwdot[CH]/d[H] */
    J[28] += dqdci;               /* dwdot[HCO]/d[H] */
    /* d()/d[OH] */
    dqdci =  + k_f*sc[9];
    J[61] += dqdci;               /* dwdot[H]/d[OH] */
    J[64] -= dqdci;               /* dwdot[OH]/d[OH] */
    J[69] -= dqdci;               /* dwdot[CH]/d[OH] */
    J[73] += dqdci;               /* dwdot[HCO]/d[OH] */
    /* d()/d[CH] */
    dqdci =  + k_f*sc[4];
    J[136] += dqdci;              /* dwdot[H]/d[CH] */
    J[139] -= dqdci;              /* dwdot[OH]/d[CH] */
    J[144] -= dqdci;              /* dwdot[CH]/d[CH] */
    J[148] += dqdci;              /* dwdot[HCO]/d[CH] */
    /* d()/d[HCO] */
    dqdci =  - k_r*sc[1];
    J[196] += dqdci;              /* dwdot[H]/d[HCO] */
    J[199] -= dqdci;              /* dwdot[OH]/d[HCO] */
    J[204] -= dqdci;              /* dwdot[CH]/d[HCO] */
    J[208] += dqdci;              /* dwdot[HCO]/d[HCO] */
    /* d()/dT */
    J[211] += dqdT;               /* dwdot[H]/dT */
    J[214] -= dqdT;               /* dwdot[OH]/dT */
    J[219] -= dqdT;               /* dwdot[CH]/dT */
    J[223] += dqdT;               /* dwdot[HCO]/dT */

    double c_R[14], dcRdT[14], e_RT[14];
    double * eh_RT;
    if (consP) {
        cp_R(c_R, tc);
        dcvpRdT(dcRdT, tc);
        eh_RT = &h_RT[0];
    }
    else {
        cv_R(c_R, tc);
        dcvpRdT(dcRdT, tc);
        speciesInternalEnergy(e_RT, tc);
        eh_RT = &e_RT[0];
    }

    double cmix = 0.0, ehmix = 0.0, dcmixdT=0.0, dehmixdT=0.0;
    for (int k = 0; k < 14; ++k) {
        cmix += c_R[k]*sc[k];
        dcmixdT += dcRdT[k]*sc[k];
        ehmix += eh_RT[k]*wdot[k];
        dehmixdT += invT*(c_R[k]-eh_RT[k])*wdot[k] + eh_RT[k]*J[210+k];
    }

    double cmixinv = 1.0/cmix;
    double tmp1 = ehmix*cmixinv;
    double tmp3 = cmixinv*T;
    double tmp2 = tmp1*tmp3;
    double dehmixdc;
    /* dTdot/d[X] */
    for (int k = 0; k < 14; ++k) {
        dehmixdc = 0.0;
        for (int m = 0; m < 14; ++m) {
            dehmixdc += eh_RT[m]*J[k*15+m];
        }
        J[k*15+14] = tmp2*c_R[k] - tmp3*dehmixdc;
    }
    /* dTdot/dT */
    J[224] = -tmp1 + tmp2*dcmixdT - tmp3*dehmixdT;

return;
}
#endif


#ifndef AMREX_USE_CUDA
/*compute the reaction Jacobian on CPU */
void aJacobian(double *  J, double *  sc, double T, int consP)
{
    for (int i=0; i<225; i++) {
        J[i] = 0.0;
    }

    double wdot[14];
    for (int k=0; k<14; k++) {
        wdot[k] = 0.0;
    }

    double tc[] = { log(T), T, T*T, T*T*T, T*T*T*T }; /*temperature cache */
    double invT = 1.0 / tc[1];
    double invT2 = invT * invT;

    /*reference concentration: P_atm / (RT) in inverse mol/m^3 */
    double refC = 101325 / 8.31446 / T;
    double refCinv = 1.0 / refC;

    /*compute the mixture concentration */
    double mixture = 0.0;
    for (int k = 0; k < 14; ++k) {
        mixture += sc[k];
    }

    /*compute the Gibbs free energy */
    double g_RT[14];
    gibbs(g_RT, tc);

    /*compute the species enthalpy */
    double h_RT[14];
    speciesEnthalpy(h_RT, tc);

    double phi_f, k_f, k_r, phi_r, Kc, q, q_nocor, Corr, alpha;
    double dlnkfdT, dlnk0dT, dlnKcdT, dkrdT, dqdT;
    double dqdci, dcdc_fac, dqdc[14];
    double Pr, fPr, F, k_0, logPr;
    double logFcent, troe_c, troe_n, troePr_den, troePr, troe;
    double Fcent1, Fcent2, Fcent3, Fcent;
    double dlogFdc, dlogFdn, dlogFdcn_fac;
    double dlogPrdT, dlogfPrdT, dlogFdT, dlogFcentdT, dlogFdlogPr, dlnCorrdT;
    const double ln10 = log(10.0);
    const double log10e = 1.0/log(10.0);
    /*reaction 1: O + HO2 => OH + O2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[2]*sc[6];
    k_f = prefactor_units[0] * fwd_A[0]
                * exp(fwd_beta[0] * tc[0] - activation_units[0] * fwd_Ea[0] * invT);
    dlnkfdT = fwd_beta[0] * invT + activation_units[0] * fwd_Ea[0] * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[2] -= q; /* O */
    wdot[3] += q; /* O2 */
    wdot[4] += q; /* OH */
    wdot[6] -= q; /* HO2 */
    /* d()/d[O] */
    dqdci =  + k_f*sc[6];
    J[32] -= dqdci;               /* dwdot[O]/d[O] */
    J[33] += dqdci;               /* dwdot[O2]/d[O] */
    J[34] += dqdci;               /* dwdot[OH]/d[O] */
    J[36] -= dqdci;               /* dwdot[HO2]/d[O] */
    /* d()/d[HO2] */
    dqdci =  + k_f*sc[2];
    J[92] -= dqdci;               /* dwdot[O]/d[HO2] */
    J[93] += dqdci;               /* dwdot[O2]/d[HO2] */
    J[94] += dqdci;               /* dwdot[OH]/d[HO2] */
    J[96] -= dqdci;               /* dwdot[HO2]/d[HO2] */
    /* d()/dT */
    J[212] -= dqdT;               /* dwdot[O]/dT */
    J[213] += dqdT;               /* dwdot[O2]/dT */
    J[214] += dqdT;               /* dwdot[OH]/dT */
    J[216] -= dqdT;               /* dwdot[HO2]/dT */

    /*reaction 2: H + HO2 => O + H2O */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[6];
    k_f = prefactor_units[1] * fwd_A[1]
                * exp(fwd_beta[1] * tc[0] - activation_units[1] * fwd_Ea[1] * invT);
    dlnkfdT = fwd_beta[1] * invT + activation_units[1] * fwd_Ea[1] * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[1] -= q; /* H */
    wdot[2] += q; /* O */
    wdot[5] += q; /* H2O */
    wdot[6] -= q; /* HO2 */
    /* d()/d[H] */
    dqdci =  + k_f*sc[6];
    J[16] -= dqdci;               /* dwdot[H]/d[H] */
    J[17] += dqdci;               /* dwdot[O]/d[H] */
    J[20] += dqdci;               /* dwdot[H2O]/d[H] */
    J[21] -= dqdci;               /* dwdot[HO2]/d[H] */
    /* d()/d[HO2] */
    dqdci =  + k_f*sc[1];
    J[91] -= dqdci;               /* dwdot[H]/d[HO2] */
    J[92] += dqdci;               /* dwdot[O]/d[HO2] */
    J[95] += dqdci;               /* dwdot[H2O]/d[HO2] */
    J[96] -= dqdci;               /* dwdot[HO2]/d[HO2] */
    /* d()/dT */
    J[211] -= dqdT;               /* dwdot[H]/dT */
    J[212] += dqdT;               /* dwdot[O]/dT */
    J[215] += dqdT;               /* dwdot[H2O]/dT */
    J[216] -= dqdT;               /* dwdot[HO2]/dT */

    /*reaction 3: H + H2O2 <=> OH + H2O */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[7];
    k_f = prefactor_units[2] * fwd_A[2]
                * exp(fwd_beta[2] * tc[0] - activation_units[2] * fwd_Ea[2] * invT);
    dlnkfdT = fwd_beta[2] * invT + activation_units[2] * fwd_Ea[2] * invT2;
    /* reverse */
    phi_r = sc[4]*sc[5];
    Kc = exp(g_RT[1] - g_RT[4] - g_RT[5] + g_RT[7]);
    k_r = k_f / Kc;
    dlnKcdT = invT * (-(h_RT[1] + h_RT[7]) + (h_RT[4] + h_RT[5]));
    dkrdT = (dlnkfdT - dlnKcdT)*k_r;
    /* rate of progress */
    q = k_f*phi_f - k_r*phi_r;
    dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
    /* update wdot */
    wdot[1] -= q; /* H */
    wdot[4] += q; /* OH */
    wdot[5] += q; /* H2O */
    wdot[7] -= q; /* H2O2 */
    /* d()/d[H] */
    dqdci =  + k_f*sc[7];
    J[16] -= dqdci;               /* dwdot[H]/d[H] */
    J[19] += dqdci;               /* dwdot[OH]/d[H] */
    J[20] += dqdci;               /* dwdot[H2O]/d[H] */
    J[22] -= dqdci;               /* dwdot[H2O2]/d[H] */
    /* d()/d[OH] */
    dqdci =  - k_r*sc[5];
    J[61] -= dqdci;               /* dwdot[H]/d[OH] */
    J[64] += dqdci;               /* dwdot[OH]/d[OH] */
    J[65] += dqdci;               /* dwdot[H2O]/d[OH] */
    J[67] -= dqdci;               /* dwdot[H2O2]/d[OH] */
    /* d()/d[H2O] */
    dqdci =  - k_r*sc[4];
    J[76] -= dqdci;               /* dwdot[H]/d[H2O] */
    J[79] += dqdci;               /* dwdot[OH]/d[H2O] */
    J[80] += dqdci;               /* dwdot[H2O]/d[H2O] */
    J[82] -= dqdci;               /* dwdot[H2O2]/d[H2O] */
    /* d()/d[H2O2] */
    dqdci =  + k_f*sc[1];
    J[106] -= dqdci;              /* dwdot[H]/d[H2O2] */
    J[109] += dqdci;              /* dwdot[OH]/d[H2O2] */
    J[110] += dqdci;              /* dwdot[H2O]/d[H2O2] */
    J[112] -= dqdci;              /* dwdot[H2O2]/d[H2O2] */
    /* d()/dT */
    J[211] -= dqdT;               /* dwdot[H]/dT */
    J[214] += dqdT;               /* dwdot[OH]/dT */
    J[215] += dqdT;               /* dwdot[H2O]/dT */
    J[217] -= dqdT;               /* dwdot[H2O2]/dT */

    /*reaction 4: O + CH => H + CO */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[2]*sc[9];
    k_f = prefactor_units[3] * fwd_A[3]
                * exp(fwd_beta[3] * tc[0] - activation_units[3] * fwd_Ea[3] * invT);
    dlnkfdT = fwd_beta[3] * invT + activation_units[3] * fwd_Ea[3] * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[1] += q; /* H */
    wdot[2] -= q; /* O */
    wdot[9] -= q; /* CH */
    wdot[11] += q; /* CO */
    /* d()/d[O] */
    dqdci =  + k_f*sc[9];
    J[31] += dqdci;               /* dwdot[H]/d[O] */
    J[32] -= dqdci;               /* dwdot[O]/d[O] */
    J[39] -= dqdci;               /* dwdot[CH]/d[O] */
    J[41] += dqdci;               /* dwdot[CO]/d[O] */
    /* d()/d[CH] */
    dqdci =  + k_f*sc[2];
    J[136] += dqdci;              /* dwdot[H]/d[CH] */
    J[137] -= dqdci;              /* dwdot[O]/d[CH] */
    J[144] -= dqdci;              /* dwdot[CH]/d[CH] */
    J[146] += dqdci;              /* dwdot[CO]/d[CH] */
    /* d()/dT */
    J[211] += dqdT;               /* dwdot[H]/dT */
    J[212] -= dqdT;               /* dwdot[O]/dT */
    J[219] -= dqdT;               /* dwdot[CH]/dT */
    J[221] += dqdT;               /* dwdot[CO]/dT */

    /*reaction 5: H + CH <=> C + H2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[9];
    k_f = prefactor_units[4] * fwd_A[4]
                * exp(fwd_beta[4] * tc[0] - activation_units[4] * fwd_Ea[4] * invT);
    dlnkfdT = fwd_beta[4] * invT + activation_units[4] * fwd_Ea[4] * invT2;
    /* reverse */
    phi_r = sc[0]*sc[8];
    Kc = exp(-g_RT[0] + g_RT[1] - g_RT[8] + g_RT[9]);
    k_r = k_f / Kc;
    dlnKcdT = invT * (-(h_RT[1] + h_RT[9]) + (h_RT[0] + h_RT[8]));
    dkrdT = (dlnkfdT - dlnKcdT)*k_r;
    /* rate of progress */
    q = k_f*phi_f - k_r*phi_r;
    dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
    /* update wdot */
    wdot[0] += q; /* H2 */
    wdot[1] -= q; /* H */
    wdot[8] += q; /* C */
    wdot[9] -= q; /* CH */
    /* d()/d[H2] */
    dqdci =  - k_r*sc[8];
    J[0] += dqdci;                /* dwdot[H2]/d[H2] */
    J[1] -= dqdci;                /* dwdot[H]/d[H2] */
    J[8] += dqdci;                /* dwdot[C]/d[H2] */
    J[9] -= dqdci;                /* dwdot[CH]/d[H2] */
    /* d()/d[H] */
    dqdci =  + k_f*sc[9];
    J[15] += dqdci;               /* dwdot[H2]/d[H] */
    J[16] -= dqdci;               /* dwdot[H]/d[H] */
    J[23] += dqdci;               /* dwdot[C]/d[H] */
    J[24] -= dqdci;               /* dwdot[CH]/d[H] */
    /* d()/d[C] */
    dqdci =  - k_r*sc[0];
    J[120] += dqdci;              /* dwdot[H2]/d[C] */
    J[121] -= dqdci;              /* dwdot[H]/d[C] */
    J[128] += dqdci;              /* dwdot[C]/d[C] */
    J[129] -= dqdci;              /* dwdot[CH]/d[C] */
    /* d()/d[CH] */
    dqdci =  + k_f*sc[1];
    J[135] += dqdci;              /* dwdot[H2]/d[CH] */
    J[136] -= dqdci;              /* dwdot[H]/d[CH] */
    J[143] += dqdci;              /* dwdot[C]/d[CH] */
    J[144] -= dqdci;              /* dwdot[CH]/d[CH] */
    /* d()/dT */
    J[210] += dqdT;               /* dwdot[H2]/dT */
    J[211] -= dqdT;               /* dwdot[H]/dT */
    J[218] += dqdT;               /* dwdot[C]/dT */
    J[219] -= dqdT;               /* dwdot[CH]/dT */

    /*reaction 6: O + CH2 <=> H + HCO */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[2]*sc[10];
    k_f = prefactor_units[5] * fwd_A[5]
                * exp(fwd_beta[5] * tc[0] - activation_units[5] * fwd_Ea[5] * invT);
    dlnkfdT = fwd_beta[5] * invT + activation_units[5] * fwd_Ea[5] * invT2;
    /* reverse */
    phi_r = sc[1]*sc[13];
    Kc = exp(-g_RT[1] + g_RT[2] + g_RT[10] - g_RT[13]);
    k_r = k_f / Kc;
    dlnKcdT = invT * (-(h_RT[2] + h_RT[10]) + (h_RT[1] + h_RT[13]));
    dkrdT = (dlnkfdT - dlnKcdT)*k_r;
    /* rate of progress */
    q = k_f*phi_f - k_r*phi_r;
    dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
    /* update wdot */
    wdot[1] += q; /* H */
    wdot[2] -= q; /* O */
    wdot[10] -= q; /* CH2 */
    wdot[13] += q; /* HCO */
    /* d()/d[H] */
    dqdci =  - k_r*sc[13];
    J[16] += dqdci;               /* dwdot[H]/d[H] */
    J[17] -= dqdci;               /* dwdot[O]/d[H] */
    J[25] -= dqdci;               /* dwdot[CH2]/d[H] */
    J[28] += dqdci;               /* dwdot[HCO]/d[H] */
    /* d()/d[O] */
    dqdci =  + k_f*sc[10];
    J[31] += dqdci;               /* dwdot[H]/d[O] */
    J[32] -= dqdci;               /* dwdot[O]/d[O] */
    J[40] -= dqdci;               /* dwdot[CH2]/d[O] */
    J[43] += dqdci;               /* dwdot[HCO]/d[O] */
    /* d()/d[CH2] */
    dqdci =  + k_f*sc[2];
    J[151] += dqdci;              /* dwdot[H]/d[CH2] */
    J[152] -= dqdci;              /* dwdot[O]/d[CH2] */
    J[160] -= dqdci;              /* dwdot[CH2]/d[CH2] */
    J[163] += dqdci;              /* dwdot[HCO]/d[CH2] */
    /* d()/d[HCO] */
    dqdci =  - k_r*sc[1];
    J[196] += dqdci;              /* dwdot[H]/d[HCO] */
    J[197] -= dqdci;              /* dwdot[O]/d[HCO] */
    J[205] -= dqdci;              /* dwdot[CH2]/d[HCO] */
    J[208] += dqdci;              /* dwdot[HCO]/d[HCO] */
    /* d()/dT */
    J[211] += dqdT;               /* dwdot[H]/dT */
    J[212] -= dqdT;               /* dwdot[O]/dT */
    J[220] -= dqdT;               /* dwdot[CH2]/dT */
    J[223] += dqdT;               /* dwdot[HCO]/dT */

    /*reaction 7: H + O2 <=> O + OH */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[3];
    k_f = prefactor_units[6] * fwd_A[6]
                * exp(fwd_beta[6] * tc[0] - activation_units[6] * fwd_Ea[6] * invT);
    dlnkfdT = fwd_beta[6] * invT + activation_units[6] * fwd_Ea[6] * invT2;
    /* reverse */
    phi_r = sc[2]*sc[4];
    Kc = exp(g_RT[1] - g_RT[2] + g_RT[3] - g_RT[4]);
    k_r = k_f / Kc;
    dlnKcdT = invT * (-(h_RT[1] + h_RT[3]) + (h_RT[2] + h_RT[4]));
    dkrdT = (dlnkfdT - dlnKcdT)*k_r;
    /* rate of progress */
    q = k_f*phi_f - k_r*phi_r;
    dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
    /* update wdot */
    wdot[1] -= q; /* H */
    wdot[2] += q; /* O */
    wdot[3] -= q; /* O2 */
    wdot[4] += q; /* OH */
    /* d()/d[H] */
    dqdci =  + k_f*sc[3];
    J[16] -= dqdci;               /* dwdot[H]/d[H] */
    J[17] += dqdci;               /* dwdot[O]/d[H] */
    J[18] -= dqdci;               /* dwdot[O2]/d[H] */
    J[19] += dqdci;               /* dwdot[OH]/d[H] */
    /* d()/d[O] */
    dqdci =  - k_r*sc[4];
    J[31] -= dqdci;               /* dwdot[H]/d[O] */
    J[32] += dqdci;               /* dwdot[O]/d[O] */
    J[33] -= dqdci;               /* dwdot[O2]/d[O] */
    J[34] += dqdci;               /* dwdot[OH]/d[O] */
    /* d()/d[O2] */
    dqdci =  + k_f*sc[1];
    J[46] -= dqdci;               /* dwdot[H]/d[O2] */
    J[47] += dqdci;               /* dwdot[O]/d[O2] */
    J[48] -= dqdci;               /* dwdot[O2]/d[O2] */
    J[49] += dqdci;               /* dwdot[OH]/d[O2] */
    /* d()/d[OH] */
    dqdci =  - k_r*sc[2];
    J[61] -= dqdci;               /* dwdot[H]/d[OH] */
    J[62] += dqdci;               /* dwdot[O]/d[OH] */
    J[63] -= dqdci;               /* dwdot[O2]/d[OH] */
    J[64] += dqdci;               /* dwdot[OH]/d[OH] */
    /* d()/dT */
    J[211] -= dqdT;               /* dwdot[H]/dT */
    J[212] += dqdT;               /* dwdot[O]/dT */
    J[213] -= dqdT;               /* dwdot[O2]/dT */
    J[214] += dqdT;               /* dwdot[OH]/dT */

    /*reaction 8: H + HO2 <=> 2.000000 OH */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[6];
    k_f = prefactor_units[7] * fwd_A[7]
                * exp(fwd_beta[7] * tc[0] - activation_units[7] * fwd_Ea[7] * invT);
    dlnkfdT = fwd_beta[7] * invT + activation_units[7] * fwd_Ea[7] * invT2;
    /* reverse */
    phi_r = pow(sc[4], 2.000000);
    Kc = exp(g_RT[1] - 2.000000*g_RT[4] + g_RT[6]);
    k_r = k_f / Kc;
    dlnKcdT = invT * (-(h_RT[1] + h_RT[6]) + (2.000000*h_RT[4]));
    dkrdT = (dlnkfdT - dlnKcdT)*k_r;
    /* rate of progress */
    q = k_f*phi_f - k_r*phi_r;
    dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
    /* update wdot */
    wdot[1] -= q; /* H */
    wdot[4] += 2 * q; /* OH */
    wdot[6] -= q; /* HO2 */
    /* d()/d[H] */
    dqdci =  + k_f*sc[6];
    J[16] -= dqdci;               /* dwdot[H]/d[H] */
    J[19] += 2 * dqdci;           /* dwdot[OH]/d[H] */
    J[21] -= dqdci;               /* dwdot[HO2]/d[H] */
    /* d()/d[OH] */
    dqdci =  - k_r*2.000000*sc[4];
    J[61] -= dqdci;               /* dwdot[H]/d[OH] */
    J[64] += 2 * dqdci;           /* dwdot[OH]/d[OH] */
    J[66] -= dqdci;               /* dwdot[HO2]/d[OH] */
    /* d()/d[HO2] */
    dqdci =  + k_f*sc[1];
    J[91] -= dqdci;               /* dwdot[H]/d[HO2] */
    J[94] += 2 * dqdci;           /* dwdot[OH]/d[HO2] */
    J[96] -= dqdci;               /* dwdot[HO2]/d[HO2] */
    /* d()/dT */
    J[211] -= dqdT;               /* dwdot[H]/dT */
    J[214] += 2 * dqdT;           /* dwdot[OH]/dT */
    J[216] -= dqdT;               /* dwdot[HO2]/dT */

    /*reaction 9: OH + CO <=> H + CO2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[4]*sc[11];
    k_f = prefactor_units[8] * fwd_A[8]
                * exp(fwd_beta[8] * tc[0] - activation_units[8] * fwd_Ea[8] * invT);
    dlnkfdT = fwd_beta[8] * invT + activation_units[8] * fwd_Ea[8] * invT2;
    /* reverse */
    phi_r = sc[1]*sc[12];
    Kc = exp(-g_RT[1] + g_RT[4] + g_RT[11] - g_RT[12]);
    k_r = k_f / Kc;
    dlnKcdT = invT * (-(h_RT[4] + h_RT[11]) + (h_RT[1] + h_RT[12]));
    dkrdT = (dlnkfdT - dlnKcdT)*k_r;
    /* rate of progress */
    q = k_f*phi_f - k_r*phi_r;
    dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
    /* update wdot */
    wdot[1] += q; /* H */
    wdot[4] -= q; /* OH */
    wdot[11] -= q; /* CO */
    wdot[12] += q; /* CO2 */
    /* d()/d[H] */
    dqdci =  - k_r*sc[12];
    J[16] += dqdci;               /* dwdot[H]/d[H] */
    J[19] -= dqdci;               /* dwdot[OH]/d[H] */
    J[26] -= dqdci;               /* dwdot[CO]/d[H] */
    J[27] += dqdci;               /* dwdot[CO2]/d[H] */
    /* d()/d[OH] */
    dqdci =  + k_f*sc[11];
    J[61] += dqdci;               /* dwdot[H]/d[OH] */
    J[64] -= dqdci;               /* dwdot[OH]/d[OH] */
    J[71] -= dqdci;               /* dwdot[CO]/d[OH] */
    J[72] += dqdci;               /* dwdot[CO2]/d[OH] */
    /* d()/d[CO] */
    dqdci =  + k_f*sc[4];
    J[166] += dqdci;              /* dwdot[H]/d[CO] */
    J[169] -= dqdci;              /* dwdot[OH]/d[CO] */
    J[176] -= dqdci;              /* dwdot[CO]/d[CO] */
    J[177] += dqdci;              /* dwdot[CO2]/d[CO] */
    /* d()/d[CO2] */
    dqdci =  - k_r*sc[1];
    J[181] += dqdci;              /* dwdot[H]/d[CO2] */
    J[184] -= dqdci;              /* dwdot[OH]/d[CO2] */
    J[191] -= dqdci;              /* dwdot[CO]/d[CO2] */
    J[192] += dqdci;              /* dwdot[CO2]/d[CO2] */
    /* d()/dT */
    J[211] += dqdT;               /* dwdot[H]/dT */
    J[214] -= dqdT;               /* dwdot[OH]/dT */
    J[221] -= dqdT;               /* dwdot[CO]/dT */
    J[222] += dqdT;               /* dwdot[CO2]/dT */

    /*reaction 10: OH + CH <=> H + HCO */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[4]*sc[9];
    k_f = prefactor_units[9] * fwd_A[9]
                * exp(fwd_beta[9] * tc[0] - activation_units[9] * fwd_Ea[9] * invT);
    dlnkfdT = fwd_beta[9] * invT + activation_units[9] * fwd_Ea[9] * invT2;
    /* reverse */
    phi_r = sc[1]*sc[13];
    Kc = exp(-g_RT[1] + g_RT[4] + g_RT[9] - g_RT[13]);
    k_r = k_f / Kc;
    dlnKcdT = invT * (-(h_RT[4] + h_RT[9]) + (h_RT[1] + h_RT[13]));
    dkrdT = (dlnkfdT - dlnKcdT)*k_r;
    /* rate of progress */
    q = k_f*phi_f - k_r*phi_r;
    dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
    /* update wdot */
    wdot[1] += q; /* H */
    wdot[4] -= q; /* OH */
    wdot[9] -= q; /* CH */
    wdot[13] += q; /* HCO */
    /* d()/d[H] */
    dqdci =  - k_r*sc[13];
    J[16] += dqdci;               /* dwdot[H]/d[H] */
    J[19] -= dqdci;               /* dwdot[OH]/d[H] */
    J[24] -= dqdci;               /* dwdot[CH]/d[H] */
    J[28] += dqdci;               /* dwdot[HCO]/d[H] */
    /* d()/d[OH] */
    dqdci =  + k_f*sc[9];
    J[61] += dqdci;               /* dwdot[H]/d[OH] */
    J[64] -= dqdci;               /* dwdot[OH]/d[OH] */
    J[69] -= dqdci;               /* dwdot[CH]/d[OH] */
    J[73] += dqdci;               /* dwdot[HCO]/d[OH] */
    /* d()/d[CH] */
    dqdci =  + k_f*sc[4];
    J[136] += dqdci;              /* dwdot[H]/d[CH] */
    J[139] -= dqdci;              /* dwdot[OH]/d[CH] */
    J[144] -= dqdci;              /* dwdot[CH]/d[CH] */
    J[148] += dqdci;              /* dwdot[HCO]/d[CH] */
    /* d()/d[HCO] */
    dqdci =  - k_r*sc[1];
    J[196] += dqdci;              /* dwdot[H]/d[HCO] */
    J[199] -= dqdci;              /* dwdot[OH]/d[HCO] */
    J[204] -= dqdci;              /* dwdot[CH]/d[HCO] */
    J[208] += dqdci;              /* dwdot[HCO]/d[HCO] */
    /* d()/dT */
    J[211] += dqdT;               /* dwdot[H]/dT */
    J[214] -= dqdT;               /* dwdot[OH]/dT */
    J[219] -= dqdT;               /* dwdot[CH]/dT */
    J[223] += dqdT;               /* dwdot[HCO]/dT */

    double c_R[14], dcRdT[14], e_RT[14];
    double * eh_RT;
    if (consP) {
        cp_R(c_R, tc);
        dcvpRdT(dcRdT, tc);
        eh_RT = &h_RT[0];
    }
    else {
        cv_R(c_R, tc);
        dcvpRdT(dcRdT, tc);
        speciesInternalEnergy(e_RT, tc);
        eh_RT = &e_RT[0];
    }

    double cmix = 0.0, ehmix = 0.0, dcmixdT=0.0, dehmixdT=0.0;
    for (int k = 0; k < 14; ++k) {
        cmix += c_R[k]*sc[k];
        dcmixdT += dcRdT[k]*sc[k];
        ehmix += eh_RT[k]*wdot[k];
        dehmixdT += invT*(c_R[k]-eh_RT[k])*wdot[k] + eh_RT[k]*J[210+k];
    }

    double cmixinv = 1.0/cmix;
    double tmp1 = ehmix*cmixinv;
    double tmp3 = cmixinv*T;
    double tmp2 = tmp1*tmp3;
    double dehmixdc;
    /* dTdot/d[X] */
    for (int k = 0; k < 14; ++k) {
        dehmixdc = 0.0;
        for (int m = 0; m < 14; ++m) {
            dehmixdc += eh_RT[m]*J[k*15+m];
        }
        J[k*15+14] = tmp2*c_R[k] - tmp3*dehmixdc;
    }
    /* dTdot/dT */
    J[224] = -tmp1 + tmp2*dcmixdT - tmp3*dehmixdT;
}
#endif


/*compute an approx to the reaction Jacobian */
AMREX_GPU_HOST_DEVICE void aJacobian_precond(double *  J, double *  sc, double T, int HP)
{
    for (int i=0; i<225; i++) {
        J[i] = 0.0;
    }

    double wdot[14];
    for (int k=0; k<14; k++) {
        wdot[k] = 0.0;
    }

    double tc[] = { log(T), T, T*T, T*T*T, T*T*T*T }; /*temperature cache */
    double invT = 1.0 / tc[1];
    double invT2 = invT * invT;

    /*reference concentration: P_atm / (RT) in inverse mol/m^3 */
    double refC = 101325 / 8.31446 / T;
    double refCinv = 1.0 / refC;

    /*compute the mixture concentration */
    double mixture = 0.0;
    for (int k = 0; k < 14; ++k) {
        mixture += sc[k];
    }

    /*compute the Gibbs free energy */
    double g_RT[14];
    gibbs(g_RT, tc);

    /*compute the species enthalpy */
    double h_RT[14];
    speciesEnthalpy(h_RT, tc);

    double phi_f, k_f, k_r, phi_r, Kc, q, q_nocor, Corr, alpha;
    double dlnkfdT, dlnk0dT, dlnKcdT, dkrdT, dqdT;
    double dqdci, dcdc_fac, dqdc[14];
    double Pr, fPr, F, k_0, logPr;
    double logFcent, troe_c, troe_n, troePr_den, troePr, troe;
    double Fcent1, Fcent2, Fcent3, Fcent;
    double dlogFdc, dlogFdn, dlogFdcn_fac;
    double dlogPrdT, dlogfPrdT, dlogFdT, dlogFcentdT, dlogFdlogPr, dlnCorrdT;
    const double ln10 = log(10.0);
    const double log10e = 1.0/log(10.0);
    /*reaction 1: O + HO2 => OH + O2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[2]*sc[6];
    k_f = 1.0000000000000002e-06 * 20000000000000
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  (0)  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[2] -= q; /* O */
    wdot[3] += q; /* O2 */
    wdot[4] += q; /* OH */
    wdot[6] -= q; /* HO2 */
    /* d()/d[O] */
    dqdci =  + k_f*sc[6];
    J[32] -= dqdci;               /* dwdot[O]/d[O] */
    J[33] += dqdci;               /* dwdot[O2]/d[O] */
    J[34] += dqdci;               /* dwdot[OH]/d[O] */
    J[36] -= dqdci;               /* dwdot[HO2]/d[O] */
    /* d()/d[HO2] */
    dqdci =  + k_f*sc[2];
    J[92] -= dqdci;               /* dwdot[O]/d[HO2] */
    J[93] += dqdci;               /* dwdot[O2]/d[HO2] */
    J[94] += dqdci;               /* dwdot[OH]/d[HO2] */
    J[96] -= dqdci;               /* dwdot[HO2]/d[HO2] */
    /* d()/dT */
    J[212] -= dqdT;               /* dwdot[O]/dT */
    J[213] += dqdT;               /* dwdot[O2]/dT */
    J[214] += dqdT;               /* dwdot[OH]/dT */
    J[216] -= dqdT;               /* dwdot[HO2]/dT */

    /*reaction 2: H + HO2 => O + H2O */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[6];
    k_f = 1.0000000000000002e-06 * 3970000000000
                * exp(0 * tc[0] - 0.50321666580471969 * (671) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  (671)  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[1] -= q; /* H */
    wdot[2] += q; /* O */
    wdot[5] += q; /* H2O */
    wdot[6] -= q; /* HO2 */
    /* d()/d[H] */
    dqdci =  + k_f*sc[6];
    J[16] -= dqdci;               /* dwdot[H]/d[H] */
    J[17] += dqdci;               /* dwdot[O]/d[H] */
    J[20] += dqdci;               /* dwdot[H2O]/d[H] */
    J[21] -= dqdci;               /* dwdot[HO2]/d[H] */
    /* d()/d[HO2] */
    dqdci =  + k_f*sc[1];
    J[91] -= dqdci;               /* dwdot[H]/d[HO2] */
    J[92] += dqdci;               /* dwdot[O]/d[HO2] */
    J[95] += dqdci;               /* dwdot[H2O]/d[HO2] */
    J[96] -= dqdci;               /* dwdot[HO2]/d[HO2] */
    /* d()/dT */
    J[211] -= dqdT;               /* dwdot[H]/dT */
    J[212] += dqdT;               /* dwdot[O]/dT */
    J[215] += dqdT;               /* dwdot[H2O]/dT */
    J[216] -= dqdT;               /* dwdot[HO2]/dT */

    /*reaction 3: H + H2O2 <=> OH + H2O */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[7];
    k_f = 1.0000000000000002e-06 * 10000000000000
                * exp(0 * tc[0] - 0.50321666580471969 * (3600) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  (3600)  * invT2;
    /* reverse */
    phi_r = sc[4]*sc[5];
    Kc = exp(g_RT[1] - g_RT[4] - g_RT[5] + g_RT[7]);
    k_r = k_f / Kc;
    dlnKcdT = invT * (-(h_RT[1] + h_RT[7]) + (h_RT[4] + h_RT[5]));
    dkrdT = (dlnkfdT - dlnKcdT)*k_r;
    /* rate of progress */
    q = k_f*phi_f - k_r*phi_r;
    dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
    /* update wdot */
    wdot[1] -= q; /* H */
    wdot[4] += q; /* OH */
    wdot[5] += q; /* H2O */
    wdot[7] -= q; /* H2O2 */
    /* d()/d[H] */
    dqdci =  + k_f*sc[7];
    J[16] -= dqdci;               /* dwdot[H]/d[H] */
    J[19] += dqdci;               /* dwdot[OH]/d[H] */
    J[20] += dqdci;               /* dwdot[H2O]/d[H] */
    J[22] -= dqdci;               /* dwdot[H2O2]/d[H] */
    /* d()/d[OH] */
    dqdci =  - k_r*sc[5];
    J[61] -= dqdci;               /* dwdot[H]/d[OH] */
    J[64] += dqdci;               /* dwdot[OH]/d[OH] */
    J[65] += dqdci;               /* dwdot[H2O]/d[OH] */
    J[67] -= dqdci;               /* dwdot[H2O2]/d[OH] */
    /* d()/d[H2O] */
    dqdci =  - k_r*sc[4];
    J[76] -= dqdci;               /* dwdot[H]/d[H2O] */
    J[79] += dqdci;               /* dwdot[OH]/d[H2O] */
    J[80] += dqdci;               /* dwdot[H2O]/d[H2O] */
    J[82] -= dqdci;               /* dwdot[H2O2]/d[H2O] */
    /* d()/d[H2O2] */
    dqdci =  + k_f*sc[1];
    J[106] -= dqdci;              /* dwdot[H]/d[H2O2] */
    J[109] += dqdci;              /* dwdot[OH]/d[H2O2] */
    J[110] += dqdci;              /* dwdot[H2O]/d[H2O2] */
    J[112] -= dqdci;              /* dwdot[H2O2]/d[H2O2] */
    /* d()/dT */
    J[211] -= dqdT;               /* dwdot[H]/dT */
    J[214] += dqdT;               /* dwdot[OH]/dT */
    J[215] += dqdT;               /* dwdot[H2O]/dT */
    J[217] -= dqdT;               /* dwdot[H2O2]/dT */

    /*reaction 4: O + CH => H + CO */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[2]*sc[9];
    k_f = 1.0000000000000002e-06 * 57000000000000
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  (0)  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[1] += q; /* H */
    wdot[2] -= q; /* O */
    wdot[9] -= q; /* CH */
    wdot[11] += q; /* CO */
    /* d()/d[O] */
    dqdci =  + k_f*sc[9];
    J[31] += dqdci;               /* dwdot[H]/d[O] */
    J[32] -= dqdci;               /* dwdot[O]/d[O] */
    J[39] -= dqdci;               /* dwdot[CH]/d[O] */
    J[41] += dqdci;               /* dwdot[CO]/d[O] */
    /* d()/d[CH] */
    dqdci =  + k_f*sc[2];
    J[136] += dqdci;              /* dwdot[H]/d[CH] */
    J[137] -= dqdci;              /* dwdot[O]/d[CH] */
    J[144] -= dqdci;              /* dwdot[CH]/d[CH] */
    J[146] += dqdci;              /* dwdot[CO]/d[CH] */
    /* d()/dT */
    J[211] += dqdT;               /* dwdot[H]/dT */
    J[212] -= dqdT;               /* dwdot[O]/dT */
    J[219] -= dqdT;               /* dwdot[CH]/dT */
    J[221] += dqdT;               /* dwdot[CO]/dT */

    /*reaction 5: H + CH <=> C + H2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[9];
    k_f = 1.0000000000000002e-06 * 110000000000000
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  (0)  * invT2;
    /* reverse */
    phi_r = sc[0]*sc[8];
    Kc = exp(-g_RT[0] + g_RT[1] - g_RT[8] + g_RT[9]);
    k_r = k_f / Kc;
    dlnKcdT = invT * (-(h_RT[1] + h_RT[9]) + (h_RT[0] + h_RT[8]));
    dkrdT = (dlnkfdT - dlnKcdT)*k_r;
    /* rate of progress */
    q = k_f*phi_f - k_r*phi_r;
    dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
    /* update wdot */
    wdot[0] += q; /* H2 */
    wdot[1] -= q; /* H */
    wdot[8] += q; /* C */
    wdot[9] -= q; /* CH */
    /* d()/d[H2] */
    dqdci =  - k_r*sc[8];
    J[0] += dqdci;                /* dwdot[H2]/d[H2] */
    J[1] -= dqdci;                /* dwdot[H]/d[H2] */
    J[8] += dqdci;                /* dwdot[C]/d[H2] */
    J[9] -= dqdci;                /* dwdot[CH]/d[H2] */
    /* d()/d[H] */
    dqdci =  + k_f*sc[9];
    J[15] += dqdci;               /* dwdot[H2]/d[H] */
    J[16] -= dqdci;               /* dwdot[H]/d[H] */
    J[23] += dqdci;               /* dwdot[C]/d[H] */
    J[24] -= dqdci;               /* dwdot[CH]/d[H] */
    /* d()/d[C] */
    dqdci =  - k_r*sc[0];
    J[120] += dqdci;              /* dwdot[H2]/d[C] */
    J[121] -= dqdci;              /* dwdot[H]/d[C] */
    J[128] += dqdci;              /* dwdot[C]/d[C] */
    J[129] -= dqdci;              /* dwdot[CH]/d[C] */
    /* d()/d[CH] */
    dqdci =  + k_f*sc[1];
    J[135] += dqdci;              /* dwdot[H2]/d[CH] */
    J[136] -= dqdci;              /* dwdot[H]/d[CH] */
    J[143] += dqdci;              /* dwdot[C]/d[CH] */
    J[144] -= dqdci;              /* dwdot[CH]/d[CH] */
    /* d()/dT */
    J[210] += dqdT;               /* dwdot[H2]/dT */
    J[211] -= dqdT;               /* dwdot[H]/dT */
    J[218] += dqdT;               /* dwdot[C]/dT */
    J[219] -= dqdT;               /* dwdot[CH]/dT */

    /*reaction 6: O + CH2 <=> H + HCO */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[2]*sc[10];
    k_f = 1.0000000000000002e-06 * 80000000000000
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  (0)  * invT2;
    /* reverse */
    phi_r = sc[1]*sc[13];
    Kc = exp(-g_RT[1] + g_RT[2] + g_RT[10] - g_RT[13]);
    k_r = k_f / Kc;
    dlnKcdT = invT * (-(h_RT[2] + h_RT[10]) + (h_RT[1] + h_RT[13]));
    dkrdT = (dlnkfdT - dlnKcdT)*k_r;
    /* rate of progress */
    q = k_f*phi_f - k_r*phi_r;
    dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
    /* update wdot */
    wdot[1] += q; /* H */
    wdot[2] -= q; /* O */
    wdot[10] -= q; /* CH2 */
    wdot[13] += q; /* HCO */
    /* d()/d[H] */
    dqdci =  - k_r*sc[13];
    J[16] += dqdci;               /* dwdot[H]/d[H] */
    J[17] -= dqdci;               /* dwdot[O]/d[H] */
    J[25] -= dqdci;               /* dwdot[CH2]/d[H] */
    J[28] += dqdci;               /* dwdot[HCO]/d[H] */
    /* d()/d[O] */
    dqdci =  + k_f*sc[10];
    J[31] += dqdci;               /* dwdot[H]/d[O] */
    J[32] -= dqdci;               /* dwdot[O]/d[O] */
    J[40] -= dqdci;               /* dwdot[CH2]/d[O] */
    J[43] += dqdci;               /* dwdot[HCO]/d[O] */
    /* d()/d[CH2] */
    dqdci =  + k_f*sc[2];
    J[151] += dqdci;              /* dwdot[H]/d[CH2] */
    J[152] -= dqdci;              /* dwdot[O]/d[CH2] */
    J[160] -= dqdci;              /* dwdot[CH2]/d[CH2] */
    J[163] += dqdci;              /* dwdot[HCO]/d[CH2] */
    /* d()/d[HCO] */
    dqdci =  - k_r*sc[1];
    J[196] += dqdci;              /* dwdot[H]/d[HCO] */
    J[197] -= dqdci;              /* dwdot[O]/d[HCO] */
    J[205] -= dqdci;              /* dwdot[CH2]/d[HCO] */
    J[208] += dqdci;              /* dwdot[HCO]/d[HCO] */
    /* d()/dT */
    J[211] += dqdT;               /* dwdot[H]/dT */
    J[212] -= dqdT;               /* dwdot[O]/dT */
    J[220] -= dqdT;               /* dwdot[CH2]/dT */
    J[223] += dqdT;               /* dwdot[HCO]/dT */

    /*reaction 7: H + O2 <=> O + OH */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[3];
    k_f = 1.0000000000000002e-06 * 83000000000000
                * exp(0 * tc[0] - 0.50321666580471969 * (14413) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  (14413)  * invT2;
    /* reverse */
    phi_r = sc[2]*sc[4];
    Kc = exp(g_RT[1] - g_RT[2] + g_RT[3] - g_RT[4]);
    k_r = k_f / Kc;
    dlnKcdT = invT * (-(h_RT[1] + h_RT[3]) + (h_RT[2] + h_RT[4]));
    dkrdT = (dlnkfdT - dlnKcdT)*k_r;
    /* rate of progress */
    q = k_f*phi_f - k_r*phi_r;
    dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
    /* update wdot */
    wdot[1] -= q; /* H */
    wdot[2] += q; /* O */
    wdot[3] -= q; /* O2 */
    wdot[4] += q; /* OH */
    /* d()/d[H] */
    dqdci =  + k_f*sc[3];
    J[16] -= dqdci;               /* dwdot[H]/d[H] */
    J[17] += dqdci;               /* dwdot[O]/d[H] */
    J[18] -= dqdci;               /* dwdot[O2]/d[H] */
    J[19] += dqdci;               /* dwdot[OH]/d[H] */
    /* d()/d[O] */
    dqdci =  - k_r*sc[4];
    J[31] -= dqdci;               /* dwdot[H]/d[O] */
    J[32] += dqdci;               /* dwdot[O]/d[O] */
    J[33] -= dqdci;               /* dwdot[O2]/d[O] */
    J[34] += dqdci;               /* dwdot[OH]/d[O] */
    /* d()/d[O2] */
    dqdci =  + k_f*sc[1];
    J[46] -= dqdci;               /* dwdot[H]/d[O2] */
    J[47] += dqdci;               /* dwdot[O]/d[O2] */
    J[48] -= dqdci;               /* dwdot[O2]/d[O2] */
    J[49] += dqdci;               /* dwdot[OH]/d[O2] */
    /* d()/d[OH] */
    dqdci =  - k_r*sc[2];
    J[61] -= dqdci;               /* dwdot[H]/d[OH] */
    J[62] += dqdci;               /* dwdot[O]/d[OH] */
    J[63] -= dqdci;               /* dwdot[O2]/d[OH] */
    J[64] += dqdci;               /* dwdot[OH]/d[OH] */
    /* d()/dT */
    J[211] -= dqdT;               /* dwdot[H]/dT */
    J[212] += dqdT;               /* dwdot[O]/dT */
    J[213] -= dqdT;               /* dwdot[O2]/dT */
    J[214] += dqdT;               /* dwdot[OH]/dT */

    /*reaction 8: H + HO2 <=> 2.000000 OH */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[6];
    k_f = 1.0000000000000002e-06 * 134000000000000
                * exp(0 * tc[0] - 0.50321666580471969 * (635) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  (635)  * invT2;
    /* reverse */
    phi_r = pow(sc[4], 2.000000);
    Kc = exp(g_RT[1] - 2.000000*g_RT[4] + g_RT[6]);
    k_r = k_f / Kc;
    dlnKcdT = invT * (-(h_RT[1] + h_RT[6]) + (2.000000*h_RT[4]));
    dkrdT = (dlnkfdT - dlnKcdT)*k_r;
    /* rate of progress */
    q = k_f*phi_f - k_r*phi_r;
    dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
    /* update wdot */
    wdot[1] -= q; /* H */
    wdot[4] += 2 * q; /* OH */
    wdot[6] -= q; /* HO2 */
    /* d()/d[H] */
    dqdci =  + k_f*sc[6];
    J[16] -= dqdci;               /* dwdot[H]/d[H] */
    J[19] += 2 * dqdci;           /* dwdot[OH]/d[H] */
    J[21] -= dqdci;               /* dwdot[HO2]/d[H] */
    /* d()/d[OH] */
    dqdci =  - k_r*2.000000*sc[4];
    J[61] -= dqdci;               /* dwdot[H]/d[OH] */
    J[64] += 2 * dqdci;           /* dwdot[OH]/d[OH] */
    J[66] -= dqdci;               /* dwdot[HO2]/d[OH] */
    /* d()/d[HO2] */
    dqdci =  + k_f*sc[1];
    J[91] -= dqdci;               /* dwdot[H]/d[HO2] */
    J[94] += 2 * dqdci;           /* dwdot[OH]/d[HO2] */
    J[96] -= dqdci;               /* dwdot[HO2]/d[HO2] */
    /* d()/dT */
    J[211] -= dqdT;               /* dwdot[H]/dT */
    J[214] += 2 * dqdT;           /* dwdot[OH]/dT */
    J[216] -= dqdT;               /* dwdot[HO2]/dT */

    /*reaction 9: OH + CO <=> H + CO2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[4]*sc[11];
    k_f = 1.0000000000000002e-06 * 47600000
                * exp(1.228 * tc[0] - 0.50321666580471969 * (70) * invT);
    dlnkfdT = 1.228 * invT + 0.50321666580471969 *  (70)  * invT2;
    /* reverse */
    phi_r = sc[1]*sc[12];
    Kc = exp(-g_RT[1] + g_RT[4] + g_RT[11] - g_RT[12]);
    k_r = k_f / Kc;
    dlnKcdT = invT * (-(h_RT[4] + h_RT[11]) + (h_RT[1] + h_RT[12]));
    dkrdT = (dlnkfdT - dlnKcdT)*k_r;
    /* rate of progress */
    q = k_f*phi_f - k_r*phi_r;
    dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
    /* update wdot */
    wdot[1] += q; /* H */
    wdot[4] -= q; /* OH */
    wdot[11] -= q; /* CO */
    wdot[12] += q; /* CO2 */
    /* d()/d[H] */
    dqdci =  - k_r*sc[12];
    J[16] += dqdci;               /* dwdot[H]/d[H] */
    J[19] -= dqdci;               /* dwdot[OH]/d[H] */
    J[26] -= dqdci;               /* dwdot[CO]/d[H] */
    J[27] += dqdci;               /* dwdot[CO2]/d[H] */
    /* d()/d[OH] */
    dqdci =  + k_f*sc[11];
    J[61] += dqdci;               /* dwdot[H]/d[OH] */
    J[64] -= dqdci;               /* dwdot[OH]/d[OH] */
    J[71] -= dqdci;               /* dwdot[CO]/d[OH] */
    J[72] += dqdci;               /* dwdot[CO2]/d[OH] */
    /* d()/d[CO] */
    dqdci =  + k_f*sc[4];
    J[166] += dqdci;              /* dwdot[H]/d[CO] */
    J[169] -= dqdci;              /* dwdot[OH]/d[CO] */
    J[176] -= dqdci;              /* dwdot[CO]/d[CO] */
    J[177] += dqdci;              /* dwdot[CO2]/d[CO] */
    /* d()/d[CO2] */
    dqdci =  - k_r*sc[1];
    J[181] += dqdci;              /* dwdot[H]/d[CO2] */
    J[184] -= dqdci;              /* dwdot[OH]/d[CO2] */
    J[191] -= dqdci;              /* dwdot[CO]/d[CO2] */
    J[192] += dqdci;              /* dwdot[CO2]/d[CO2] */
    /* d()/dT */
    J[211] += dqdT;               /* dwdot[H]/dT */
    J[214] -= dqdT;               /* dwdot[OH]/dT */
    J[221] -= dqdT;               /* dwdot[CO]/dT */
    J[222] += dqdT;               /* dwdot[CO2]/dT */

    /*reaction 10: OH + CH <=> H + HCO */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[4]*sc[9];
    k_f = 1.0000000000000002e-06 * 30000000000000
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  (0)  * invT2;
    /* reverse */
    phi_r = sc[1]*sc[13];
    Kc = exp(-g_RT[1] + g_RT[4] + g_RT[9] - g_RT[13]);
    k_r = k_f / Kc;
    dlnKcdT = invT * (-(h_RT[4] + h_RT[9]) + (h_RT[1] + h_RT[13]));
    dkrdT = (dlnkfdT - dlnKcdT)*k_r;
    /* rate of progress */
    q = k_f*phi_f - k_r*phi_r;
    dqdT = (dlnkfdT*k_f*phi_f - dkrdT*phi_r);
    /* update wdot */
    wdot[1] += q; /* H */
    wdot[4] -= q; /* OH */
    wdot[9] -= q; /* CH */
    wdot[13] += q; /* HCO */
    /* d()/d[H] */
    dqdci =  - k_r*sc[13];
    J[16] += dqdci;               /* dwdot[H]/d[H] */
    J[19] -= dqdci;               /* dwdot[OH]/d[H] */
    J[24] -= dqdci;               /* dwdot[CH]/d[H] */
    J[28] += dqdci;               /* dwdot[HCO]/d[H] */
    /* d()/d[OH] */
    dqdci =  + k_f*sc[9];
    J[61] += dqdci;               /* dwdot[H]/d[OH] */
    J[64] -= dqdci;               /* dwdot[OH]/d[OH] */
    J[69] -= dqdci;               /* dwdot[CH]/d[OH] */
    J[73] += dqdci;               /* dwdot[HCO]/d[OH] */
    /* d()/d[CH] */
    dqdci =  + k_f*sc[4];
    J[136] += dqdci;              /* dwdot[H]/d[CH] */
    J[139] -= dqdci;              /* dwdot[OH]/d[CH] */
    J[144] -= dqdci;              /* dwdot[CH]/d[CH] */
    J[148] += dqdci;              /* dwdot[HCO]/d[CH] */
    /* d()/d[HCO] */
    dqdci =  - k_r*sc[1];
    J[196] += dqdci;              /* dwdot[H]/d[HCO] */
    J[199] -= dqdci;              /* dwdot[OH]/d[HCO] */
    J[204] -= dqdci;              /* dwdot[CH]/d[HCO] */
    J[208] += dqdci;              /* dwdot[HCO]/d[HCO] */
    /* d()/dT */
    J[211] += dqdT;               /* dwdot[H]/dT */
    J[214] -= dqdT;               /* dwdot[OH]/dT */
    J[219] -= dqdT;               /* dwdot[CH]/dT */
    J[223] += dqdT;               /* dwdot[HCO]/dT */

    double c_R[14], dcRdT[14], e_RT[14];
    double * eh_RT;
    if (HP) {
        cp_R(c_R, tc);
        dcvpRdT(dcRdT, tc);
        eh_RT = &h_RT[0];
    }
    else {
        cv_R(c_R, tc);
        dcvpRdT(dcRdT, tc);
        speciesInternalEnergy(e_RT, tc);
        eh_RT = &e_RT[0];
    }

    double cmix = 0.0, ehmix = 0.0, dcmixdT=0.0, dehmixdT=0.0;
    for (int k = 0; k < 14; ++k) {
        cmix += c_R[k]*sc[k];
        dcmixdT += dcRdT[k]*sc[k];
        ehmix += eh_RT[k]*wdot[k];
        dehmixdT += invT*(c_R[k]-eh_RT[k])*wdot[k] + eh_RT[k]*J[210+k];
    }

    double cmixinv = 1.0/cmix;
    double tmp1 = ehmix*cmixinv;
    double tmp3 = cmixinv*T;
    double tmp2 = tmp1*tmp3;
    double dehmixdc;
    /* dTdot/d[X] */
    for (int k = 0; k < 14; ++k) {
        dehmixdc = 0.0;
        for (int m = 0; m < 14; ++m) {
            dehmixdc += eh_RT[m]*J[k*15+m];
        }
        J[k*15+14] = tmp2*c_R[k] - tmp3*dehmixdc;
    }
    /* dTdot/dT */
    J[224] = -tmp1 + tmp2*dcmixdT - tmp3*dehmixdT;
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
        /*species 3: O2 */
        species[3] =
            -2.99673416e-03
            +1.96946040e-05 * tc[1]
            -2.90438853e-08 * tc[2]
            +1.29749135e-11 * tc[3];
        /*species 4: OH */
        species[4] =
            -2.40131752e-03
            +9.23587682e-06 * tc[1]
            -1.16434000e-08 * tc[2]
            +5.45645880e-12 * tc[3];
        /*species 5: H2O */
        species[5] =
            -2.03643410e-03
            +1.30408042e-05 * tc[1]
            -1.64639119e-08 * tc[2]
            +7.08791268e-12 * tc[3];
        /*species 6: HO2 */
        species[6] =
            -4.74912051e-03
            +4.23165782e-05 * tc[1]
            -7.28291682e-08 * tc[2]
            +3.71690050e-11 * tc[3];
        /*species 7: H2O2 */
        species[7] =
            -5.42822417e-04
            +3.34671402e-05 * tc[1]
            -6.47312439e-08 * tc[2]
            +3.44981745e-11 * tc[3];
        /*species 8: C */
        species[8] =
            -3.21537724e-04
            +1.46758449e-06 * tc[1]
            -2.19670467e-09 * tc[2]
            +1.06608578e-12 * tc[3];
        /*species 9: CH */
        species[9] =
            +3.23835541e-04
            -3.37798130e-06 * tc[1]
            +9.48651981e-09 * tc[2]
            -5.62436268e-12 * tc[3];
        /*species 10: CH2 */
        species[10] =
            +9.68872143e-04
            +5.58979682e-06 * tc[1]
            -1.15527346e-08 * tc[2]
            +6.74966876e-12 * tc[3];
        /*species 11: CO */
        species[11] =
            -6.10353680e-04
            +2.03362866e-06 * tc[1]
            +2.72101765e-09 * tc[2]
            -3.61769800e-12 * tc[3];
        /*species 12: CO2 */
        species[12] =
            +8.98459677e-03
            -1.42471254e-05 * tc[1]
            +7.37757066e-09 * tc[2]
            -5.74798192e-13 * tc[3];
        /*species 13: HCO */
        species[13] =
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
        /*species 3: O2 */
        species[3] =
            +1.48308754e-03
            -1.51593334e-06 * tc[1]
            +6.28411665e-10 * tc[2]
            -8.66871176e-14 * tc[3];
        /*species 4: OH */
        species[4] =
            +5.48429716e-04
            +2.53010456e-07 * tc[1]
            -2.63838467e-10 * tc[2]
            +4.69649504e-14 * tc[3];
        /*species 5: H2O */
        species[5] =
            +2.17691804e-03
            -3.28145036e-07 * tc[1]
            -2.91125961e-10 * tc[2]
            +6.72803968e-14 * tc[3];
        /*species 6: HO2 */
        species[6] =
            +2.23982013e-03
            -1.26731630e-06 * tc[1]
            +3.42739110e-10 * tc[2]
            -4.31634140e-14 * tc[3];
        /*species 7: H2O2 */
        species[7] =
            +4.90831694e-03
            -3.80278450e-06 * tc[1]
            +1.11355796e-09 * tc[2]
            -1.15163322e-13 * tc[3];
        /*species 8: C */
        species[8] =
            +4.79889284e-05
            -1.44867004e-07 * tc[1]
            +1.12287309e-10 * tc[2]
            -1.94911157e-14 * tc[3];
        /*species 9: CH */
        species[9] =
            +9.70913681e-04
            +2.88891310e-07 * tc[1]
            -3.92063547e-10 * tc[2]
            +7.04317532e-14 * tc[3];
        /*species 10: CH2 */
        species[10] =
            +3.65639292e-03
            -2.81789194e-06 * tc[1]
            +7.80538647e-10 * tc[2]
            -7.50910268e-14 * tc[3];
        /*species 11: CO */
        species[11] =
            +2.06252743e-03
            -1.99765154e-06 * tc[1]
            +6.90159024e-10 * tc[2]
            -8.14590864e-14 * tc[3];
        /*species 12: CO2 */
        species[12] =
            +4.41437026e-03
            -4.42962808e-06 * tc[1]
            +1.57047056e-09 * tc[2]
            -1.88833666e-13 * tc[3];
        /*species 13: HCO */
        species[13] =
            +4.95695526e-03
            -4.96891226e-06 * tc[1]
            +1.76748533e-09 * tc[2]
            -2.13403484e-13 * tc[3];
    }
    return;
}


/*compute the progress rate for each reaction */
AMREX_GPU_HOST_DEVICE void progressRate(double *  qdot, double *  sc, double T)
{
    double tc[] = { log(T), T, T*T, T*T*T, T*T*T*T }; /*temperature cache */
    double invT = 1.0 / tc[1];

#ifndef AMREX_USE_CUDA
    if (T != T_save)
    {
        T_save = T;
        comp_k_f(tc,invT,k_f_save);
        comp_Kc(tc,invT,Kc_save);
    }
#endif

    double q_f[10], q_r[10];
    comp_qfqr(q_f, q_r, sc, tc, invT);

    for (int i = 0; i < 10; ++i) {
        qdot[i] = q_f[i] - q_r[i];
    }

    return;
}


/*compute the progress rate for each reaction */
AMREX_GPU_HOST_DEVICE void progressRateFR(double *  q_f, double *  q_r, double *  sc, double T)
{
    double tc[] = { log(T), T, T*T, T*T*T, T*T*T*T }; /*temperature cache */
    double invT = 1.0 / tc[1];
#ifndef AMREX_USE_CUDA

    if (T != T_save)
    {
        T_save = T;
        comp_k_f(tc,invT,k_f_save);
        comp_Kc(tc,invT,Kc_save);
    }
#endif

    comp_qfqr(q_f, q_r, sc, tc, invT);

    return;
}


/*compute the equilibrium constants for each reaction */
void equilibriumConstants(double *  kc, double *  g_RT, double T)
{
    /*reference concentration: P_atm / (RT) in inverse mol/m^3 */
    double refC = 101325 / 8.31446 / T;

    /*reaction 1: O + HO2 => OH + O2 */
    kc[0] = exp((g_RT[2] + g_RT[6]) - (g_RT[4] + g_RT[3]));

    /*reaction 2: H + HO2 => O + H2O */
    kc[1] = exp((g_RT[1] + g_RT[6]) - (g_RT[2] + g_RT[5]));

    /*reaction 3: H + H2O2 <=> OH + H2O */
    kc[2] = exp((g_RT[1] + g_RT[7]) - (g_RT[4] + g_RT[5]));

    /*reaction 4: O + CH => H + CO */
    kc[3] = exp((g_RT[2] + g_RT[9]) - (g_RT[1] + g_RT[11]));

    /*reaction 5: H + CH <=> C + H2 */
    kc[4] = exp((g_RT[1] + g_RT[9]) - (g_RT[8] + g_RT[0]));

    /*reaction 6: O + CH2 <=> H + HCO */
    kc[5] = exp((g_RT[2] + g_RT[10]) - (g_RT[1] + g_RT[13]));

    /*reaction 7: H + O2 <=> O + OH */
    kc[6] = exp((g_RT[1] + g_RT[3]) - (g_RT[2] + g_RT[4]));

    /*reaction 8: H + HO2 <=> 2.000000 OH */
    kc[7] = exp((g_RT[1] + g_RT[6]) - (2.000000 * g_RT[4]));

    /*reaction 9: OH + CO <=> H + CO2 */
    kc[8] = exp((g_RT[4] + g_RT[11]) - (g_RT[1] + g_RT[12]));

    /*reaction 10: OH + CH <=> H + HCO */
    kc[9] = exp((g_RT[4] + g_RT[9]) - (g_RT[1] + g_RT[13]));

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
        /*species 3: O2 */
        species[3] =
            -1.063943560000000e+03 * invT
            +1.247806300000001e-01
            -3.782456360000000e+00 * tc[0]
            +1.498367080000000e-03 * tc[1]
            -1.641217001666667e-06 * tc[2]
            +8.067745908333334e-10 * tc[3]
            -1.621864185000000e-13 * tc[4];
        /*species 4: OH */
        species[4] =
            +3.615080560000000e+03 * invT
            +4.095940888000000e+00
            -3.992015430000000e+00 * tc[0]
            +1.200658760000000e-03 * tc[1]
            -7.696564016666666e-07 * tc[2]
            +3.234277775000000e-10 * tc[3]
            -6.820573500000000e-14 * tc[4];
        /*species 5: H2O */
        species[5] =
            -3.029372670000000e+04 * invT
            +5.047672768000000e+00
            -4.198640560000000e+00 * tc[0]
            +1.018217050000000e-03 * tc[1]
            -1.086733685000000e-06 * tc[2]
            +4.573308850000000e-10 * tc[3]
            -8.859890850000000e-14 * tc[4];
        /*species 6: HO2 */
        species[6] =
            +2.948080400000000e+02 * invT
            +5.851355599999999e-01
            -4.301798010000000e+00 * tc[0]
            +2.374560255000000e-03 * tc[1]
            -3.526381516666666e-06 * tc[2]
            +2.023032450000000e-09 * tc[3]
            -4.646125620000001e-13 * tc[4];
        /*species 7: H2O2 */
        species[7] =
            -1.770258210000000e+04 * invT
            +8.410619499999998e-01
            -4.276112690000000e+00 * tc[0]
            +2.714112085000000e-04 * tc[1]
            -2.788928350000000e-06 * tc[2]
            +1.798090108333333e-09 * tc[3]
            -4.312271815000000e-13 * tc[4];
        /*species 8: C */
        species[8] =
            +8.544388320000000e+04 * invT
            -1.977068930000000e+00
            -2.554239550000000e+00 * tc[0]
            +1.607688620000000e-04 * tc[1]
            -1.222987075000000e-07 * tc[2]
            +6.101957408333333e-11 * tc[3]
            -1.332607230000000e-14 * tc[4];
        /*species 9: CH */
        species[9] =
            +7.079729340000000e+04 * invT
            +1.405805570000000e+00
            -3.489816650000000e+00 * tc[0]
            -1.619177705000000e-04 * tc[1]
            +2.814984416666667e-07 * tc[2]
            -2.635144391666666e-10 * tc[3]
            +7.030453350000001e-14 * tc[4];
        /*species 10: CH2 */
        species[10] =
            +4.600404010000000e+04 * invT
            +2.200146820000000e+00
            -3.762678670000000e+00 * tc[0]
            -4.844360715000000e-04 * tc[1]
            -4.658164016666667e-07 * tc[2]
            +3.209092941666667e-10 * tc[3]
            -8.437085950000000e-14 * tc[4];
        /*species 11: CO */
        species[11] =
            -1.434408600000000e+04 * invT
            +7.112418999999992e-02
            -3.579533470000000e+00 * tc[0]
            +3.051768400000000e-04 * tc[1]
            -1.694690550000000e-07 * tc[2]
            -7.558382366666667e-11 * tc[3]
            +4.522122495000000e-14 * tc[4];
        /*species 12: CO2 */
        species[12] =
            -4.837196970000000e+04 * invT
            -7.544278700000000e+00
            -2.356773520000000e+00 * tc[0]
            -4.492298385000000e-03 * tc[1]
            +1.187260448333333e-06 * tc[2]
            -2.049325183333333e-10 * tc[3]
            +7.184977399999999e-15 * tc[4];
        /*species 13: HCO */
        species[13] =
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
        /*species 3: O2 */
        species[3] =
            -1.088457720000000e+03 * invT
            -2.170693450000000e+00
            -3.282537840000000e+00 * tc[0]
            -7.415437700000000e-04 * tc[1]
            +1.263277781666667e-07 * tc[2]
            -1.745587958333333e-11 * tc[3]
            +1.083588970000000e-15 * tc[4];
        /*species 4: OH */
        species[4] =
            +3.858657000000000e+03 * invT
            -1.383808430000000e+00
            -3.092887670000000e+00 * tc[0]
            -2.742148580000000e-04 * tc[1]
            -2.108420466666667e-08 * tc[2]
            +7.328846300000000e-12 * tc[3]
            -5.870618800000000e-16 * tc[4];
        /*species 5: H2O */
        species[5] =
            -3.000429710000000e+04 * invT
            -1.932777610000000e+00
            -3.033992490000000e+00 * tc[0]
            -1.088459020000000e-03 * tc[1]
            +2.734541966666666e-08 * tc[2]
            +8.086832250000000e-12 * tc[3]
            -8.410049600000000e-16 * tc[4];
        /*species 6: HO2 */
        species[6] =
            +1.118567130000000e+02 * invT
            +2.321087500000001e-01
            -4.017210900000000e+00 * tc[0]
            -1.119910065000000e-03 * tc[1]
            +1.056096916666667e-07 * tc[2]
            -9.520530833333334e-12 * tc[3]
            +5.395426750000000e-16 * tc[4];
        /*species 7: H2O2 */
        species[7] =
            -1.786178770000000e+04 * invT
            +1.248846229999999e+00
            -4.165002850000000e+00 * tc[0]
            -2.454158470000000e-03 * tc[1]
            +3.168987083333333e-07 * tc[2]
            -3.093216550000000e-11 * tc[3]
            +1.439541525000000e-15 * tc[4];
        /*species 8: C */
        species[8] =
            +8.545129530000000e+04 * invT
            -2.308834850000000e+00
            -2.492668880000000e+00 * tc[0]
            -2.399446420000000e-05 * tc[1]
            +1.207225033333333e-08 * tc[2]
            -3.119091908333333e-12 * tc[3]
            +2.436389465000000e-16 * tc[4];
        /*species 9: CH */
        species[9] =
            +7.101243640000001e+04 * invT
            -2.606515260000000e+00
            -2.878464730000000e+00 * tc[0]
            -4.854568405000000e-04 * tc[1]
            -2.407427583333333e-08 * tc[2]
            +1.089065408333333e-11 * tc[3]
            -8.803969149999999e-16 * tc[4];
        /*species 10: CH2 */
        species[10] =
            +4.626360400000000e+04 * invT
            -3.297092110000000e+00
            -2.874101130000000e+00 * tc[0]
            -1.828196460000000e-03 * tc[1]
            +2.348243283333333e-07 * tc[2]
            -2.168162908333333e-11 * tc[3]
            +9.386378350000000e-16 * tc[4];
        /*species 11: CO */
        species[11] =
            -1.415187240000000e+04 * invT
            -5.103502110000000e+00
            -2.715185610000000e+00 * tc[0]
            -1.031263715000000e-03 * tc[1]
            +1.664709618333334e-07 * tc[2]
            -1.917108400000000e-11 * tc[3]
            +1.018238580000000e-15 * tc[4];
        /*species 12: CO2 */
        species[12] =
            -4.875916600000000e+04 * invT
            +1.585822230000000e+00
            -3.857460290000000e+00 * tc[0]
            -2.207185130000000e-03 * tc[1]
            +3.691356733333334e-07 * tc[2]
            -4.362418233333334e-11 * tc[3]
            +2.360420820000000e-15 * tc[4];
        /*species 13: HCO */
        species[13] =
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
        /*species 3: O2 */
        species[3] =
            -1.06394356e+03 * invT
            -8.75219370e-01
            -3.78245636e+00 * tc[0]
            +1.49836708e-03 * tc[1]
            -1.64121700e-06 * tc[2]
            +8.06774591e-10 * tc[3]
            -1.62186418e-13 * tc[4];
        /*species 4: OH */
        species[4] =
            +3.61508056e+03 * invT
            +3.09594089e+00
            -3.99201543e+00 * tc[0]
            +1.20065876e-03 * tc[1]
            -7.69656402e-07 * tc[2]
            +3.23427778e-10 * tc[3]
            -6.82057350e-14 * tc[4];
        /*species 5: H2O */
        species[5] =
            -3.02937267e+04 * invT
            +4.04767277e+00
            -4.19864056e+00 * tc[0]
            +1.01821705e-03 * tc[1]
            -1.08673369e-06 * tc[2]
            +4.57330885e-10 * tc[3]
            -8.85989085e-14 * tc[4];
        /*species 6: HO2 */
        species[6] =
            +2.94808040e+02 * invT
            -4.14864440e-01
            -4.30179801e+00 * tc[0]
            +2.37456025e-03 * tc[1]
            -3.52638152e-06 * tc[2]
            +2.02303245e-09 * tc[3]
            -4.64612562e-13 * tc[4];
        /*species 7: H2O2 */
        species[7] =
            -1.77025821e+04 * invT
            -1.58938050e-01
            -4.27611269e+00 * tc[0]
            +2.71411208e-04 * tc[1]
            -2.78892835e-06 * tc[2]
            +1.79809011e-09 * tc[3]
            -4.31227182e-13 * tc[4];
        /*species 8: C */
        species[8] =
            +8.54438832e+04 * invT
            -2.97706893e+00
            -2.55423955e+00 * tc[0]
            +1.60768862e-04 * tc[1]
            -1.22298707e-07 * tc[2]
            +6.10195741e-11 * tc[3]
            -1.33260723e-14 * tc[4];
        /*species 9: CH */
        species[9] =
            +7.07972934e+04 * invT
            +4.05805570e-01
            -3.48981665e+00 * tc[0]
            -1.61917771e-04 * tc[1]
            +2.81498442e-07 * tc[2]
            -2.63514439e-10 * tc[3]
            +7.03045335e-14 * tc[4];
        /*species 10: CH2 */
        species[10] =
            +4.60040401e+04 * invT
            +1.20014682e+00
            -3.76267867e+00 * tc[0]
            -4.84436072e-04 * tc[1]
            -4.65816402e-07 * tc[2]
            +3.20909294e-10 * tc[3]
            -8.43708595e-14 * tc[4];
        /*species 11: CO */
        species[11] =
            -1.43440860e+04 * invT
            -9.28875810e-01
            -3.57953347e+00 * tc[0]
            +3.05176840e-04 * tc[1]
            -1.69469055e-07 * tc[2]
            -7.55838237e-11 * tc[3]
            +4.52212249e-14 * tc[4];
        /*species 12: CO2 */
        species[12] =
            -4.83719697e+04 * invT
            -8.54427870e+00
            -2.35677352e+00 * tc[0]
            -4.49229839e-03 * tc[1]
            +1.18726045e-06 * tc[2]
            -2.04932518e-10 * tc[3]
            +7.18497740e-15 * tc[4];
        /*species 13: HCO */
        species[13] =
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
        /*species 3: O2 */
        species[3] =
            -1.08845772e+03 * invT
            -3.17069345e+00
            -3.28253784e+00 * tc[0]
            -7.41543770e-04 * tc[1]
            +1.26327778e-07 * tc[2]
            -1.74558796e-11 * tc[3]
            +1.08358897e-15 * tc[4];
        /*species 4: OH */
        species[4] =
            +3.85865700e+03 * invT
            -2.38380843e+00
            -3.09288767e+00 * tc[0]
            -2.74214858e-04 * tc[1]
            -2.10842047e-08 * tc[2]
            +7.32884630e-12 * tc[3]
            -5.87061880e-16 * tc[4];
        /*species 5: H2O */
        species[5] =
            -3.00042971e+04 * invT
            -2.93277761e+00
            -3.03399249e+00 * tc[0]
            -1.08845902e-03 * tc[1]
            +2.73454197e-08 * tc[2]
            +8.08683225e-12 * tc[3]
            -8.41004960e-16 * tc[4];
        /*species 6: HO2 */
        species[6] =
            +1.11856713e+02 * invT
            -7.67891250e-01
            -4.01721090e+00 * tc[0]
            -1.11991006e-03 * tc[1]
            +1.05609692e-07 * tc[2]
            -9.52053083e-12 * tc[3]
            +5.39542675e-16 * tc[4];
        /*species 7: H2O2 */
        species[7] =
            -1.78617877e+04 * invT
            +2.48846230e-01
            -4.16500285e+00 * tc[0]
            -2.45415847e-03 * tc[1]
            +3.16898708e-07 * tc[2]
            -3.09321655e-11 * tc[3]
            +1.43954153e-15 * tc[4];
        /*species 8: C */
        species[8] =
            +8.54512953e+04 * invT
            -3.30883485e+00
            -2.49266888e+00 * tc[0]
            -2.39944642e-05 * tc[1]
            +1.20722503e-08 * tc[2]
            -3.11909191e-12 * tc[3]
            +2.43638946e-16 * tc[4];
        /*species 9: CH */
        species[9] =
            +7.10124364e+04 * invT
            -3.60651526e+00
            -2.87846473e+00 * tc[0]
            -4.85456840e-04 * tc[1]
            -2.40742758e-08 * tc[2]
            +1.08906541e-11 * tc[3]
            -8.80396915e-16 * tc[4];
        /*species 10: CH2 */
        species[10] =
            +4.62636040e+04 * invT
            -4.29709211e+00
            -2.87410113e+00 * tc[0]
            -1.82819646e-03 * tc[1]
            +2.34824328e-07 * tc[2]
            -2.16816291e-11 * tc[3]
            +9.38637835e-16 * tc[4];
        /*species 11: CO */
        species[11] =
            -1.41518724e+04 * invT
            -6.10350211e+00
            -2.71518561e+00 * tc[0]
            -1.03126372e-03 * tc[1]
            +1.66470962e-07 * tc[2]
            -1.91710840e-11 * tc[3]
            +1.01823858e-15 * tc[4];
        /*species 12: CO2 */
        species[12] =
            -4.87591660e+04 * invT
            +5.85822230e-01
            -3.85746029e+00 * tc[0]
            -2.20718513e-03 * tc[1]
            +3.69135673e-07 * tc[2]
            -4.36241823e-11 * tc[3]
            +2.36042082e-15 * tc[4];
        /*species 13: HCO */
        species[13] =
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
        /*species 3: O2 */
        species[3] =
            +2.78245636e+00
            -2.99673416e-03 * tc[1]
            +9.84730201e-06 * tc[2]
            -9.68129509e-09 * tc[3]
            +3.24372837e-12 * tc[4];
        /*species 4: OH */
        species[4] =
            +2.99201543e+00
            -2.40131752e-03 * tc[1]
            +4.61793841e-06 * tc[2]
            -3.88113333e-09 * tc[3]
            +1.36411470e-12 * tc[4];
        /*species 5: H2O */
        species[5] =
            +3.19864056e+00
            -2.03643410e-03 * tc[1]
            +6.52040211e-06 * tc[2]
            -5.48797062e-09 * tc[3]
            +1.77197817e-12 * tc[4];
        /*species 6: HO2 */
        species[6] =
            +3.30179801e+00
            -4.74912051e-03 * tc[1]
            +2.11582891e-05 * tc[2]
            -2.42763894e-08 * tc[3]
            +9.29225124e-12 * tc[4];
        /*species 7: H2O2 */
        species[7] =
            +3.27611269e+00
            -5.42822417e-04 * tc[1]
            +1.67335701e-05 * tc[2]
            -2.15770813e-08 * tc[3]
            +8.62454363e-12 * tc[4];
        /*species 8: C */
        species[8] =
            +1.55423955e+00
            -3.21537724e-04 * tc[1]
            +7.33792245e-07 * tc[2]
            -7.32234889e-10 * tc[3]
            +2.66521446e-13 * tc[4];
        /*species 9: CH */
        species[9] =
            +2.48981665e+00
            +3.23835541e-04 * tc[1]
            -1.68899065e-06 * tc[2]
            +3.16217327e-09 * tc[3]
            -1.40609067e-12 * tc[4];
        /*species 10: CH2 */
        species[10] =
            +2.76267867e+00
            +9.68872143e-04 * tc[1]
            +2.79489841e-06 * tc[2]
            -3.85091153e-09 * tc[3]
            +1.68741719e-12 * tc[4];
        /*species 11: CO */
        species[11] =
            +2.57953347e+00
            -6.10353680e-04 * tc[1]
            +1.01681433e-06 * tc[2]
            +9.07005884e-10 * tc[3]
            -9.04424499e-13 * tc[4];
        /*species 12: CO2 */
        species[12] =
            +1.35677352e+00
            +8.98459677e-03 * tc[1]
            -7.12356269e-06 * tc[2]
            +2.45919022e-09 * tc[3]
            -1.43699548e-13 * tc[4];
        /*species 13: HCO */
        species[13] =
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
        /*species 3: O2 */
        species[3] =
            +2.28253784e+00
            +1.48308754e-03 * tc[1]
            -7.57966669e-07 * tc[2]
            +2.09470555e-10 * tc[3]
            -2.16717794e-14 * tc[4];
        /*species 4: OH */
        species[4] =
            +2.09288767e+00
            +5.48429716e-04 * tc[1]
            +1.26505228e-07 * tc[2]
            -8.79461556e-11 * tc[3]
            +1.17412376e-14 * tc[4];
        /*species 5: H2O */
        species[5] =
            +2.03399249e+00
            +2.17691804e-03 * tc[1]
            -1.64072518e-07 * tc[2]
            -9.70419870e-11 * tc[3]
            +1.68200992e-14 * tc[4];
        /*species 6: HO2 */
        species[6] =
            +3.01721090e+00
            +2.23982013e-03 * tc[1]
            -6.33658150e-07 * tc[2]
            +1.14246370e-10 * tc[3]
            -1.07908535e-14 * tc[4];
        /*species 7: H2O2 */
        species[7] =
            +3.16500285e+00
            +4.90831694e-03 * tc[1]
            -1.90139225e-06 * tc[2]
            +3.71185986e-10 * tc[3]
            -2.87908305e-14 * tc[4];
        /*species 8: C */
        species[8] =
            +1.49266888e+00
            +4.79889284e-05 * tc[1]
            -7.24335020e-08 * tc[2]
            +3.74291029e-11 * tc[3]
            -4.87277893e-15 * tc[4];
        /*species 9: CH */
        species[9] =
            +1.87846473e+00
            +9.70913681e-04 * tc[1]
            +1.44445655e-07 * tc[2]
            -1.30687849e-10 * tc[3]
            +1.76079383e-14 * tc[4];
        /*species 10: CH2 */
        species[10] =
            +1.87410113e+00
            +3.65639292e-03 * tc[1]
            -1.40894597e-06 * tc[2]
            +2.60179549e-10 * tc[3]
            -1.87727567e-14 * tc[4];
        /*species 11: CO */
        species[11] =
            +1.71518561e+00
            +2.06252743e-03 * tc[1]
            -9.98825771e-07 * tc[2]
            +2.30053008e-10 * tc[3]
            -2.03647716e-14 * tc[4];
        /*species 12: CO2 */
        species[12] =
            +2.85746029e+00
            +4.41437026e-03 * tc[1]
            -2.21481404e-06 * tc[2]
            +5.23490188e-10 * tc[3]
            -4.72084164e-14 * tc[4];
        /*species 13: HCO */
        species[13] =
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
        /*species 3: O2 */
        species[3] =
            +3.78245636e+00
            -2.99673416e-03 * tc[1]
            +9.84730201e-06 * tc[2]
            -9.68129509e-09 * tc[3]
            +3.24372837e-12 * tc[4];
        /*species 4: OH */
        species[4] =
            +3.99201543e+00
            -2.40131752e-03 * tc[1]
            +4.61793841e-06 * tc[2]
            -3.88113333e-09 * tc[3]
            +1.36411470e-12 * tc[4];
        /*species 5: H2O */
        species[5] =
            +4.19864056e+00
            -2.03643410e-03 * tc[1]
            +6.52040211e-06 * tc[2]
            -5.48797062e-09 * tc[3]
            +1.77197817e-12 * tc[4];
        /*species 6: HO2 */
        species[6] =
            +4.30179801e+00
            -4.74912051e-03 * tc[1]
            +2.11582891e-05 * tc[2]
            -2.42763894e-08 * tc[3]
            +9.29225124e-12 * tc[4];
        /*species 7: H2O2 */
        species[7] =
            +4.27611269e+00
            -5.42822417e-04 * tc[1]
            +1.67335701e-05 * tc[2]
            -2.15770813e-08 * tc[3]
            +8.62454363e-12 * tc[4];
        /*species 8: C */
        species[8] =
            +2.55423955e+00
            -3.21537724e-04 * tc[1]
            +7.33792245e-07 * tc[2]
            -7.32234889e-10 * tc[3]
            +2.66521446e-13 * tc[4];
        /*species 9: CH */
        species[9] =
            +3.48981665e+00
            +3.23835541e-04 * tc[1]
            -1.68899065e-06 * tc[2]
            +3.16217327e-09 * tc[3]
            -1.40609067e-12 * tc[4];
        /*species 10: CH2 */
        species[10] =
            +3.76267867e+00
            +9.68872143e-04 * tc[1]
            +2.79489841e-06 * tc[2]
            -3.85091153e-09 * tc[3]
            +1.68741719e-12 * tc[4];
        /*species 11: CO */
        species[11] =
            +3.57953347e+00
            -6.10353680e-04 * tc[1]
            +1.01681433e-06 * tc[2]
            +9.07005884e-10 * tc[3]
            -9.04424499e-13 * tc[4];
        /*species 12: CO2 */
        species[12] =
            +2.35677352e+00
            +8.98459677e-03 * tc[1]
            -7.12356269e-06 * tc[2]
            +2.45919022e-09 * tc[3]
            -1.43699548e-13 * tc[4];
        /*species 13: HCO */
        species[13] =
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
        /*species 3: O2 */
        species[3] =
            +3.28253784e+00
            +1.48308754e-03 * tc[1]
            -7.57966669e-07 * tc[2]
            +2.09470555e-10 * tc[3]
            -2.16717794e-14 * tc[4];
        /*species 4: OH */
        species[4] =
            +3.09288767e+00
            +5.48429716e-04 * tc[1]
            +1.26505228e-07 * tc[2]
            -8.79461556e-11 * tc[3]
            +1.17412376e-14 * tc[4];
        /*species 5: H2O */
        species[5] =
            +3.03399249e+00
            +2.17691804e-03 * tc[1]
            -1.64072518e-07 * tc[2]
            -9.70419870e-11 * tc[3]
            +1.68200992e-14 * tc[4];
        /*species 6: HO2 */
        species[6] =
            +4.01721090e+00
            +2.23982013e-03 * tc[1]
            -6.33658150e-07 * tc[2]
            +1.14246370e-10 * tc[3]
            -1.07908535e-14 * tc[4];
        /*species 7: H2O2 */
        species[7] =
            +4.16500285e+00
            +4.90831694e-03 * tc[1]
            -1.90139225e-06 * tc[2]
            +3.71185986e-10 * tc[3]
            -2.87908305e-14 * tc[4];
        /*species 8: C */
        species[8] =
            +2.49266888e+00
            +4.79889284e-05 * tc[1]
            -7.24335020e-08 * tc[2]
            +3.74291029e-11 * tc[3]
            -4.87277893e-15 * tc[4];
        /*species 9: CH */
        species[9] =
            +2.87846473e+00
            +9.70913681e-04 * tc[1]
            +1.44445655e-07 * tc[2]
            -1.30687849e-10 * tc[3]
            +1.76079383e-14 * tc[4];
        /*species 10: CH2 */
        species[10] =
            +2.87410113e+00
            +3.65639292e-03 * tc[1]
            -1.40894597e-06 * tc[2]
            +2.60179549e-10 * tc[3]
            -1.87727567e-14 * tc[4];
        /*species 11: CO */
        species[11] =
            +2.71518561e+00
            +2.06252743e-03 * tc[1]
            -9.98825771e-07 * tc[2]
            +2.30053008e-10 * tc[3]
            -2.03647716e-14 * tc[4];
        /*species 12: CO2 */
        species[12] =
            +3.85746029e+00
            +4.41437026e-03 * tc[1]
            -2.21481404e-06 * tc[2]
            +5.23490188e-10 * tc[3]
            -4.72084164e-14 * tc[4];
        /*species 13: HCO */
        species[13] =
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
        /*species 3: O2 */
        species[3] =
            +2.78245636e+00
            -1.49836708e-03 * tc[1]
            +3.28243400e-06 * tc[2]
            -2.42032377e-09 * tc[3]
            +6.48745674e-13 * tc[4]
            -1.06394356e+03 * invT;
        /*species 4: OH */
        species[4] =
            +2.99201543e+00
            -1.20065876e-03 * tc[1]
            +1.53931280e-06 * tc[2]
            -9.70283332e-10 * tc[3]
            +2.72822940e-13 * tc[4]
            +3.61508056e+03 * invT;
        /*species 5: H2O */
        species[5] =
            +3.19864056e+00
            -1.01821705e-03 * tc[1]
            +2.17346737e-06 * tc[2]
            -1.37199266e-09 * tc[3]
            +3.54395634e-13 * tc[4]
            -3.02937267e+04 * invT;
        /*species 6: HO2 */
        species[6] =
            +3.30179801e+00
            -2.37456025e-03 * tc[1]
            +7.05276303e-06 * tc[2]
            -6.06909735e-09 * tc[3]
            +1.85845025e-12 * tc[4]
            +2.94808040e+02 * invT;
        /*species 7: H2O2 */
        species[7] =
            +3.27611269e+00
            -2.71411208e-04 * tc[1]
            +5.57785670e-06 * tc[2]
            -5.39427032e-09 * tc[3]
            +1.72490873e-12 * tc[4]
            -1.77025821e+04 * invT;
        /*species 8: C */
        species[8] =
            +1.55423955e+00
            -1.60768862e-04 * tc[1]
            +2.44597415e-07 * tc[2]
            -1.83058722e-10 * tc[3]
            +5.33042892e-14 * tc[4]
            +8.54438832e+04 * invT;
        /*species 9: CH */
        species[9] =
            +2.48981665e+00
            +1.61917771e-04 * tc[1]
            -5.62996883e-07 * tc[2]
            +7.90543317e-10 * tc[3]
            -2.81218134e-13 * tc[4]
            +7.07972934e+04 * invT;
        /*species 10: CH2 */
        species[10] =
            +2.76267867e+00
            +4.84436072e-04 * tc[1]
            +9.31632803e-07 * tc[2]
            -9.62727883e-10 * tc[3]
            +3.37483438e-13 * tc[4]
            +4.60040401e+04 * invT;
        /*species 11: CO */
        species[11] =
            +2.57953347e+00
            -3.05176840e-04 * tc[1]
            +3.38938110e-07 * tc[2]
            +2.26751471e-10 * tc[3]
            -1.80884900e-13 * tc[4]
            -1.43440860e+04 * invT;
        /*species 12: CO2 */
        species[12] =
            +1.35677352e+00
            +4.49229839e-03 * tc[1]
            -2.37452090e-06 * tc[2]
            +6.14797555e-10 * tc[3]
            -2.87399096e-14 * tc[4]
            -4.83719697e+04 * invT;
        /*species 13: HCO */
        species[13] =
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
        /*species 3: O2 */
        species[3] =
            +2.28253784e+00
            +7.41543770e-04 * tc[1]
            -2.52655556e-07 * tc[2]
            +5.23676387e-11 * tc[3]
            -4.33435588e-15 * tc[4]
            -1.08845772e+03 * invT;
        /*species 4: OH */
        species[4] =
            +2.09288767e+00
            +2.74214858e-04 * tc[1]
            +4.21684093e-08 * tc[2]
            -2.19865389e-11 * tc[3]
            +2.34824752e-15 * tc[4]
            +3.85865700e+03 * invT;
        /*species 5: H2O */
        species[5] =
            +2.03399249e+00
            +1.08845902e-03 * tc[1]
            -5.46908393e-08 * tc[2]
            -2.42604967e-11 * tc[3]
            +3.36401984e-15 * tc[4]
            -3.00042971e+04 * invT;
        /*species 6: HO2 */
        species[6] =
            +3.01721090e+00
            +1.11991006e-03 * tc[1]
            -2.11219383e-07 * tc[2]
            +2.85615925e-11 * tc[3]
            -2.15817070e-15 * tc[4]
            +1.11856713e+02 * invT;
        /*species 7: H2O2 */
        species[7] =
            +3.16500285e+00
            +2.45415847e-03 * tc[1]
            -6.33797417e-07 * tc[2]
            +9.27964965e-11 * tc[3]
            -5.75816610e-15 * tc[4]
            -1.78617877e+04 * invT;
        /*species 8: C */
        species[8] =
            +1.49266888e+00
            +2.39944642e-05 * tc[1]
            -2.41445007e-08 * tc[2]
            +9.35727573e-12 * tc[3]
            -9.74555786e-16 * tc[4]
            +8.54512953e+04 * invT;
        /*species 9: CH */
        species[9] =
            +1.87846473e+00
            +4.85456840e-04 * tc[1]
            +4.81485517e-08 * tc[2]
            -3.26719623e-11 * tc[3]
            +3.52158766e-15 * tc[4]
            +7.10124364e+04 * invT;
        /*species 10: CH2 */
        species[10] =
            +1.87410113e+00
            +1.82819646e-03 * tc[1]
            -4.69648657e-07 * tc[2]
            +6.50448872e-11 * tc[3]
            -3.75455134e-15 * tc[4]
            +4.62636040e+04 * invT;
        /*species 11: CO */
        species[11] =
            +1.71518561e+00
            +1.03126372e-03 * tc[1]
            -3.32941924e-07 * tc[2]
            +5.75132520e-11 * tc[3]
            -4.07295432e-15 * tc[4]
            -1.41518724e+04 * invT;
        /*species 12: CO2 */
        species[12] =
            +2.85746029e+00
            +2.20718513e-03 * tc[1]
            -7.38271347e-07 * tc[2]
            +1.30872547e-10 * tc[3]
            -9.44168328e-15 * tc[4]
            -4.87591660e+04 * invT;
        /*species 13: HCO */
        species[13] =
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
        /*species 3: O2 */
        species[3] =
            +3.78245636e+00
            -1.49836708e-03 * tc[1]
            +3.28243400e-06 * tc[2]
            -2.42032377e-09 * tc[3]
            +6.48745674e-13 * tc[4]
            -1.06394356e+03 * invT;
        /*species 4: OH */
        species[4] =
            +3.99201543e+00
            -1.20065876e-03 * tc[1]
            +1.53931280e-06 * tc[2]
            -9.70283332e-10 * tc[3]
            +2.72822940e-13 * tc[4]
            +3.61508056e+03 * invT;
        /*species 5: H2O */
        species[5] =
            +4.19864056e+00
            -1.01821705e-03 * tc[1]
            +2.17346737e-06 * tc[2]
            -1.37199266e-09 * tc[3]
            +3.54395634e-13 * tc[4]
            -3.02937267e+04 * invT;
        /*species 6: HO2 */
        species[6] =
            +4.30179801e+00
            -2.37456025e-03 * tc[1]
            +7.05276303e-06 * tc[2]
            -6.06909735e-09 * tc[3]
            +1.85845025e-12 * tc[4]
            +2.94808040e+02 * invT;
        /*species 7: H2O2 */
        species[7] =
            +4.27611269e+00
            -2.71411208e-04 * tc[1]
            +5.57785670e-06 * tc[2]
            -5.39427032e-09 * tc[3]
            +1.72490873e-12 * tc[4]
            -1.77025821e+04 * invT;
        /*species 8: C */
        species[8] =
            +2.55423955e+00
            -1.60768862e-04 * tc[1]
            +2.44597415e-07 * tc[2]
            -1.83058722e-10 * tc[3]
            +5.33042892e-14 * tc[4]
            +8.54438832e+04 * invT;
        /*species 9: CH */
        species[9] =
            +3.48981665e+00
            +1.61917771e-04 * tc[1]
            -5.62996883e-07 * tc[2]
            +7.90543317e-10 * tc[3]
            -2.81218134e-13 * tc[4]
            +7.07972934e+04 * invT;
        /*species 10: CH2 */
        species[10] =
            +3.76267867e+00
            +4.84436072e-04 * tc[1]
            +9.31632803e-07 * tc[2]
            -9.62727883e-10 * tc[3]
            +3.37483438e-13 * tc[4]
            +4.60040401e+04 * invT;
        /*species 11: CO */
        species[11] =
            +3.57953347e+00
            -3.05176840e-04 * tc[1]
            +3.38938110e-07 * tc[2]
            +2.26751471e-10 * tc[3]
            -1.80884900e-13 * tc[4]
            -1.43440860e+04 * invT;
        /*species 12: CO2 */
        species[12] =
            +2.35677352e+00
            +4.49229839e-03 * tc[1]
            -2.37452090e-06 * tc[2]
            +6.14797555e-10 * tc[3]
            -2.87399096e-14 * tc[4]
            -4.83719697e+04 * invT;
        /*species 13: HCO */
        species[13] =
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
        /*species 3: O2 */
        species[3] =
            +3.28253784e+00
            +7.41543770e-04 * tc[1]
            -2.52655556e-07 * tc[2]
            +5.23676387e-11 * tc[3]
            -4.33435588e-15 * tc[4]
            -1.08845772e+03 * invT;
        /*species 4: OH */
        species[4] =
            +3.09288767e+00
            +2.74214858e-04 * tc[1]
            +4.21684093e-08 * tc[2]
            -2.19865389e-11 * tc[3]
            +2.34824752e-15 * tc[4]
            +3.85865700e+03 * invT;
        /*species 5: H2O */
        species[5] =
            +3.03399249e+00
            +1.08845902e-03 * tc[1]
            -5.46908393e-08 * tc[2]
            -2.42604967e-11 * tc[3]
            +3.36401984e-15 * tc[4]
            -3.00042971e+04 * invT;
        /*species 6: HO2 */
        species[6] =
            +4.01721090e+00
            +1.11991006e-03 * tc[1]
            -2.11219383e-07 * tc[2]
            +2.85615925e-11 * tc[3]
            -2.15817070e-15 * tc[4]
            +1.11856713e+02 * invT;
        /*species 7: H2O2 */
        species[7] =
            +4.16500285e+00
            +2.45415847e-03 * tc[1]
            -6.33797417e-07 * tc[2]
            +9.27964965e-11 * tc[3]
            -5.75816610e-15 * tc[4]
            -1.78617877e+04 * invT;
        /*species 8: C */
        species[8] =
            +2.49266888e+00
            +2.39944642e-05 * tc[1]
            -2.41445007e-08 * tc[2]
            +9.35727573e-12 * tc[3]
            -9.74555786e-16 * tc[4]
            +8.54512953e+04 * invT;
        /*species 9: CH */
        species[9] =
            +2.87846473e+00
            +4.85456840e-04 * tc[1]
            +4.81485517e-08 * tc[2]
            -3.26719623e-11 * tc[3]
            +3.52158766e-15 * tc[4]
            +7.10124364e+04 * invT;
        /*species 10: CH2 */
        species[10] =
            +2.87410113e+00
            +1.82819646e-03 * tc[1]
            -4.69648657e-07 * tc[2]
            +6.50448872e-11 * tc[3]
            -3.75455134e-15 * tc[4]
            +4.62636040e+04 * invT;
        /*species 11: CO */
        species[11] =
            +2.71518561e+00
            +1.03126372e-03 * tc[1]
            -3.32941924e-07 * tc[2]
            +5.75132520e-11 * tc[3]
            -4.07295432e-15 * tc[4]
            -1.41518724e+04 * invT;
        /*species 12: CO2 */
        species[12] =
            +3.85746029e+00
            +2.20718513e-03 * tc[1]
            -7.38271347e-07 * tc[2]
            +1.30872547e-10 * tc[3]
            -9.44168328e-15 * tc[4]
            -4.87591660e+04 * invT;
        /*species 13: HCO */
        species[13] =
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
        /*species 3: O2 */
        species[3] =
            +3.78245636e+00 * tc[0]
            -2.99673416e-03 * tc[1]
            +4.92365101e-06 * tc[2]
            -3.22709836e-09 * tc[3]
            +8.10932092e-13 * tc[4]
            +3.65767573e+00 ;
        /*species 4: OH */
        species[4] =
            +3.99201543e+00 * tc[0]
            -2.40131752e-03 * tc[1]
            +2.30896920e-06 * tc[2]
            -1.29371111e-09 * tc[3]
            +3.41028675e-13 * tc[4]
            -1.03925458e-01 ;
        /*species 5: H2O */
        species[5] =
            +4.19864056e+00 * tc[0]
            -2.03643410e-03 * tc[1]
            +3.26020105e-06 * tc[2]
            -1.82932354e-09 * tc[3]
            +4.42994543e-13 * tc[4]
            -8.49032208e-01 ;
        /*species 6: HO2 */
        species[6] =
            +4.30179801e+00 * tc[0]
            -4.74912051e-03 * tc[1]
            +1.05791445e-05 * tc[2]
            -8.09212980e-09 * tc[3]
            +2.32306281e-12 * tc[4]
            +3.71666245e+00 ;
        /*species 7: H2O2 */
        species[7] =
            +4.27611269e+00 * tc[0]
            -5.42822417e-04 * tc[1]
            +8.36678505e-06 * tc[2]
            -7.19236043e-09 * tc[3]
            +2.15613591e-12 * tc[4]
            +3.43505074e+00 ;
        /*species 8: C */
        species[8] =
            +2.55423955e+00 * tc[0]
            -3.21537724e-04 * tc[1]
            +3.66896122e-07 * tc[2]
            -2.44078296e-10 * tc[3]
            +6.66303615e-14 * tc[4]
            +4.53130848e+00 ;
        /*species 9: CH */
        species[9] =
            +3.48981665e+00 * tc[0]
            +3.23835541e-04 * tc[1]
            -8.44495325e-07 * tc[2]
            +1.05405776e-09 * tc[3]
            -3.51522668e-13 * tc[4]
            +2.08401108e+00 ;
        /*species 10: CH2 */
        species[10] =
            +3.76267867e+00 * tc[0]
            +9.68872143e-04 * tc[1]
            +1.39744921e-06 * tc[2]
            -1.28363718e-09 * tc[3]
            +4.21854298e-13 * tc[4]
            +1.56253185e+00 ;
        /*species 11: CO */
        species[11] =
            +3.57953347e+00 * tc[0]
            -6.10353680e-04 * tc[1]
            +5.08407165e-07 * tc[2]
            +3.02335295e-10 * tc[3]
            -2.26106125e-13 * tc[4]
            +3.50840928e+00 ;
        /*species 12: CO2 */
        species[12] =
            +2.35677352e+00 * tc[0]
            +8.98459677e-03 * tc[1]
            -3.56178134e-06 * tc[2]
            +8.19730073e-10 * tc[3]
            -3.59248870e-14 * tc[4]
            +9.90105222e+00 ;
        /*species 13: HCO */
        species[13] =
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
        /*species 3: O2 */
        species[3] =
            +3.28253784e+00 * tc[0]
            +1.48308754e-03 * tc[1]
            -3.78983334e-07 * tc[2]
            +6.98235183e-11 * tc[3]
            -5.41794485e-15 * tc[4]
            +5.45323129e+00 ;
        /*species 4: OH */
        species[4] =
            +3.09288767e+00 * tc[0]
            +5.48429716e-04 * tc[1]
            +6.32526140e-08 * tc[2]
            -2.93153852e-11 * tc[3]
            +2.93530940e-15 * tc[4]
            +4.47669610e+00 ;
        /*species 5: H2O */
        species[5] =
            +3.03399249e+00 * tc[0]
            +2.17691804e-03 * tc[1]
            -8.20362590e-08 * tc[2]
            -3.23473290e-11 * tc[3]
            +4.20502480e-15 * tc[4]
            +4.96677010e+00 ;
        /*species 6: HO2 */
        species[6] =
            +4.01721090e+00 * tc[0]
            +2.23982013e-03 * tc[1]
            -3.16829075e-07 * tc[2]
            +3.80821233e-11 * tc[3]
            -2.69771337e-15 * tc[4]
            +3.78510215e+00 ;
        /*species 7: H2O2 */
        species[7] =
            +4.16500285e+00 * tc[0]
            +4.90831694e-03 * tc[1]
            -9.50696125e-07 * tc[2]
            +1.23728662e-10 * tc[3]
            -7.19770763e-15 * tc[4]
            +2.91615662e+00 ;
        /*species 8: C */
        species[8] =
            +2.49266888e+00 * tc[0]
            +4.79889284e-05 * tc[1]
            -3.62167510e-08 * tc[2]
            +1.24763676e-11 * tc[3]
            -1.21819473e-15 * tc[4]
            +4.80150373e+00 ;
        /*species 9: CH */
        species[9] =
            +2.87846473e+00 * tc[0]
            +9.70913681e-04 * tc[1]
            +7.22228275e-08 * tc[2]
            -4.35626163e-11 * tc[3]
            +4.40198457e-15 * tc[4]
            +5.48497999e+00 ;
        /*species 10: CH2 */
        species[10] =
            +2.87410113e+00 * tc[0]
            +3.65639292e-03 * tc[1]
            -7.04472985e-07 * tc[2]
            +8.67265163e-11 * tc[3]
            -4.69318918e-15 * tc[4]
            +6.17119324e+00 ;
        /*species 11: CO */
        species[11] =
            +2.71518561e+00 * tc[0]
            +2.06252743e-03 * tc[1]
            -4.99412886e-07 * tc[2]
            +7.66843360e-11 * tc[3]
            -5.09119290e-15 * tc[4]
            +7.81868772e+00 ;
        /*species 12: CO2 */
        species[12] =
            +3.85746029e+00 * tc[0]
            +4.41437026e-03 * tc[1]
            -1.10740702e-06 * tc[2]
            +1.74496729e-10 * tc[3]
            -1.18021041e-14 * tc[4]
            +2.27163806e+00 ;
        /*species 13: HCO */
        species[13] =
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


/*compute the critical parameters for each species */
void GET_CRITPARAMS(double *  Tci, double *  ai, double *  bi, double *  acentric_i)
{

    double   EPS[14];
    double   SIG[14];
    double    wt[14];
    double avogadro = 6.02214199e23;
    double boltzmann = 1.3806503e-16; //we work in CGS
    double Rcst = 83.144598; //in bar [CGS] !

    egtransetEPS(EPS);
    egtransetSIG(SIG);
    get_mw(wt);

    /*species 0: H2 */
    /*Imported from NIST */
    Tci[0] = 33.145000 ; 
    ai[0] = 1e6 * 0.42748 * pow(Rcst,2.0) * pow(Tci[0],2.0) / (pow(2.015880,2.0) * 12.964000); 
    bi[0] = 0.08664 * Rcst * Tci[0] / (2.015880 * 12.964000); 
    acentric_i[0] = -0.219000 ;

    /*species 1: H */
    Tci[1] = 1.316 * EPS[1] ; 
    ai[1] = (5.55 * pow(avogadro,2.0) * EPS[1]*boltzmann * pow(1e-8*SIG[1],3.0) ) / (pow(wt[1],2.0)); 
    bi[1] = 0.855 * avogadro * pow(1e-8*SIG[1],3.0) / (wt[1]); 
    acentric_i[1] = 0.0 ;

    /*species 2: O */
    Tci[2] = 1.316 * EPS[2] ; 
    ai[2] = (5.55 * pow(avogadro,2.0) * EPS[2]*boltzmann * pow(1e-8*SIG[2],3.0) ) / (pow(wt[2],2.0)); 
    bi[2] = 0.855 * avogadro * pow(1e-8*SIG[2],3.0) / (wt[2]); 
    acentric_i[2] = 0.0 ;

    /*species 3: O2 */
    /*Imported from NIST */
    Tci[3] = 154.581000 ; 
    ai[3] = 1e6 * 0.42748 * pow(Rcst,2.0) * pow(Tci[3],2.0) / (pow(31.998800,2.0) * 50.430466); 
    bi[3] = 0.08664 * Rcst * Tci[3] / (31.998800 * 50.430466); 
    acentric_i[3] = 0.022200 ;

    /*species 4: OH */
    Tci[4] = 1.316 * EPS[4] ; 
    ai[4] = (5.55 * pow(avogadro,2.0) * EPS[4]*boltzmann * pow(1e-8*SIG[4],3.0) ) / (pow(wt[4],2.0)); 
    bi[4] = 0.855 * avogadro * pow(1e-8*SIG[4],3.0) / (wt[4]); 
    acentric_i[4] = 0.0 ;

    /*species 5: H2O */
    /*Imported from NIST */
    Tci[5] = 647.096000 ; 
    ai[5] = 1e6 * 0.42748 * pow(Rcst,2.0) * pow(Tci[5],2.0) / (pow(18.015340,2.0) * 220.640000); 
    bi[5] = 0.08664 * Rcst * Tci[5] / (18.015340 * 220.640000); 
    acentric_i[5] = 0.344300 ;

    /*species 6: HO2 */
    Tci[6] = 1.316 * EPS[6] ; 
    ai[6] = (5.55 * pow(avogadro,2.0) * EPS[6]*boltzmann * pow(1e-8*SIG[6],3.0) ) / (pow(wt[6],2.0)); 
    bi[6] = 0.855 * avogadro * pow(1e-8*SIG[6],3.0) / (wt[6]); 
    acentric_i[6] = 0.0 ;

    /*species 7: H2O2 */
    Tci[7] = 1.316 * EPS[7] ; 
    ai[7] = (5.55 * pow(avogadro,2.0) * EPS[7]*boltzmann * pow(1e-8*SIG[7],3.0) ) / (pow(wt[7],2.0)); 
    bi[7] = 0.855 * avogadro * pow(1e-8*SIG[7],3.0) / (wt[7]); 
    acentric_i[7] = 0.0 ;

    /*species 8: C */
    Tci[8] = 1.316 * EPS[8] ; 
    ai[8] = (5.55 * pow(avogadro,2.0) * EPS[8]*boltzmann * pow(1e-8*SIG[8],3.0) ) / (pow(wt[8],2.0)); 
    bi[8] = 0.855 * avogadro * pow(1e-8*SIG[8],3.0) / (wt[8]); 
    acentric_i[8] = 0.0 ;

    /*species 9: CH */
    Tci[9] = 1.316 * EPS[9] ; 
    ai[9] = (5.55 * pow(avogadro,2.0) * EPS[9]*boltzmann * pow(1e-8*SIG[9],3.0) ) / (pow(wt[9],2.0)); 
    bi[9] = 0.855 * avogadro * pow(1e-8*SIG[9],3.0) / (wt[9]); 
    acentric_i[9] = 0.0 ;

    /*species 10: CH2 */
    Tci[10] = 1.316 * EPS[10] ; 
    ai[10] = (5.55 * pow(avogadro,2.0) * EPS[10]*boltzmann * pow(1e-8*SIG[10],3.0) ) / (pow(wt[10],2.0)); 
    bi[10] = 0.855 * avogadro * pow(1e-8*SIG[10],3.0) / (wt[10]); 
    acentric_i[10] = 0.0 ;

    /*species 11: CO */
    /*Imported from NIST */
    Tci[11] = 132.850000 ; 
    ai[11] = 1e6 * 0.42748 * pow(Rcst,2.0) * pow(Tci[11],2.0) / (pow(28.010000,2.0) * 34.940000); 
    bi[11] = 0.08664 * Rcst * Tci[11] / (28.010000 * 34.940000); 
    acentric_i[11] = 0.045000 ;

    /*species 12: CO2 */
    /*Imported from NIST */
    Tci[12] = 304.120000 ; 
    ai[12] = 1e6 * 0.42748 * pow(Rcst,2.0) * pow(Tci[12],2.0) / (pow(44.009950,2.0) * 73.740000); 
    bi[12] = 0.08664 * Rcst * Tci[12] / (44.009950 * 73.740000); 
    acentric_i[12] = 0.225000 ;

    /*species 13: HCO */
    Tci[13] = 1.316 * EPS[13] ; 
    ai[13] = (5.55 * pow(avogadro,2.0) * EPS[13]*boltzmann * pow(1e-8*SIG[13],3.0) ) / (pow(wt[13],2.0)); 
    bi[13] = 0.855 * avogadro * pow(1e-8*SIG[13],3.0) / (wt[13]); 
    acentric_i[13] = 0.0 ;

    return;
}


void egtransetLENIMC(int* LENIMC ) {
    *LENIMC = 58;}


void egtransetLENRMC(int* LENRMC ) {
    *LENRMC = 4214;}


void egtransetNO(int* NO ) {
    *NO = 4;}


void egtransetKK(int* KK ) {
    *KK = 14;}


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
    WT[3] = 3.19988000E+01;
    WT[4] = 1.70073700E+01;
    WT[5] = 1.80153400E+01;
    WT[6] = 3.30067700E+01;
    WT[7] = 3.40147400E+01;
    WT[8] = 1.20111500E+01;
    WT[9] = 1.30191200E+01;
    WT[10] = 1.40270900E+01;
    WT[11] = 2.80105500E+01;
    WT[12] = 4.40099500E+01;
    WT[13] = 2.90185200E+01;
}


/*the lennard-jones potential well depth eps/kb in K */
void egtransetEPS(double* EPS ) {
    EPS[0] = 3.80000000E+01;
    EPS[1] = 1.45000000E+02;
    EPS[2] = 8.00000000E+01;
    EPS[3] = 1.07400000E+02;
    EPS[4] = 8.00000000E+01;
    EPS[5] = 5.72400000E+02;
    EPS[6] = 1.07400000E+02;
    EPS[7] = 1.07400000E+02;
    EPS[8] = 7.14000000E+01;
    EPS[9] = 8.00000000E+01;
    EPS[10] = 1.44000000E+02;
    EPS[11] = 9.81000000E+01;
    EPS[12] = 2.44000000E+02;
    EPS[13] = 4.98000000E+02;
}


/*the lennard-jones collision diameter in Angstroms */
void egtransetSIG(double* SIG ) {
    SIG[0] = 2.92000000E+00;
    SIG[1] = 2.05000000E+00;
    SIG[2] = 2.75000000E+00;
    SIG[3] = 3.45800000E+00;
    SIG[4] = 2.75000000E+00;
    SIG[5] = 2.60500000E+00;
    SIG[6] = 3.45800000E+00;
    SIG[7] = 3.45800000E+00;
    SIG[8] = 3.29800000E+00;
    SIG[9] = 2.75000000E+00;
    SIG[10] = 3.80000000E+00;
    SIG[11] = 3.65000000E+00;
    SIG[12] = 3.76300000E+00;
    SIG[13] = 3.59000000E+00;
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
}


/*the polarizability in cubic Angstroms */
void egtransetPOL(double* POL ) {
    POL[0] = 7.90000000E-01;
    POL[1] = 0.00000000E+00;
    POL[2] = 0.00000000E+00;
    POL[3] = 1.60000000E+00;
    POL[4] = 0.00000000E+00;
    POL[5] = 0.00000000E+00;
    POL[6] = 0.00000000E+00;
    POL[7] = 0.00000000E+00;
    POL[8] = 0.00000000E+00;
    POL[9] = 0.00000000E+00;
    POL[10] = 0.00000000E+00;
    POL[11] = 1.95000000E+00;
    POL[12] = 2.65000000E+00;
    POL[13] = 0.00000000E+00;
}


/*the rotational relaxation collision number at 298 K */
void egtransetZROT(double* ZROT ) {
    ZROT[0] = 2.80000000E+02;
    ZROT[1] = 0.00000000E+00;
    ZROT[2] = 0.00000000E+00;
    ZROT[3] = 3.80000000E+00;
    ZROT[4] = 0.00000000E+00;
    ZROT[5] = 4.00000000E+00;
    ZROT[6] = 1.00000000E+00;
    ZROT[7] = 3.80000000E+00;
    ZROT[8] = 0.00000000E+00;
    ZROT[9] = 0.00000000E+00;
    ZROT[10] = 0.00000000E+00;
    ZROT[11] = 1.80000000E+00;
    ZROT[12] = 2.10000000E+00;
    ZROT[13] = 0.00000000E+00;
}


/*0: monoatomic, 1: linear, 2: nonlinear */
void egtransetNLIN(int* NLIN) {
    NLIN[0] = 1;
    NLIN[1] = 0;
    NLIN[2] = 0;
    NLIN[3] = 1;
    NLIN[4] = 1;
    NLIN[5] = 2;
    NLIN[6] = 2;
    NLIN[7] = 2;
    NLIN[8] = 0;
    NLIN[9] = 1;
    NLIN[10] = 1;
    NLIN[11] = 1;
    NLIN[12] = 1;
    NLIN[13] = 2;
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
    COFETA[12] = -1.78915826E+01;
    COFETA[13] = 2.98311502E+00;
    COFETA[14] = -3.14105508E-01;
    COFETA[15] = 1.40500162E-02;
    COFETA[16] = -1.57526788E+01;
    COFETA[17] = 2.21311180E+00;
    COFETA[18] = -2.12959301E-01;
    COFETA[19] = 9.62195191E-03;
    COFETA[20] = -1.14441613E+01;
    COFETA[21] = -9.67162014E-01;
    COFETA[22] = 3.58651197E-01;
    COFETA[23] = -2.09789135E-02;
    COFETA[24] = -1.78760754E+01;
    COFETA[25] = 2.98311502E+00;
    COFETA[26] = -3.14105508E-01;
    COFETA[27] = 1.40500162E-02;
    COFETA[28] = -1.78610348E+01;
    COFETA[29] = 2.98311502E+00;
    COFETA[30] = -3.14105508E-01;
    COFETA[31] = 1.40500162E-02;
    COFETA[32] = -1.56481804E+01;
    COFETA[33] = 1.96214031E+00;
    COFETA[34] = -1.79397210E-01;
    COFETA[35] = 8.12672879E-03;
    COFETA[36] = -1.58862927E+01;
    COFETA[37] = 2.21311180E+00;
    COFETA[38] = -2.12959301E-01;
    COFETA[39] = 9.62195191E-03;
    COFETA[40] = -2.06998946E+01;
    COFETA[41] = 3.80445652E+00;
    COFETA[42] = -4.18022689E-01;
    COFETA[43] = 1.84337719E-02;
    COFETA[44] = -1.74469975E+01;
    COFETA[45] = 2.74728386E+00;
    COFETA[46] = -2.83509015E-01;
    COFETA[47] = 1.27267083E-02;
    COFETA[48] = -2.28110345E+01;
    COFETA[49] = 4.62954710E+00;
    COFETA[50] = -5.00689001E-01;
    COFETA[51] = 2.10012969E-02;
    COFETA[52] = -1.57492286E+01;
    COFETA[53] = 9.81716009E-01;
    COFETA[54] = 7.09938181E-02;
    COFETA[55] = -7.70023966E-03;
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
    COFLAM[12] = 5.31578403E-01;
    COFLAM[13] = 1.87067453E+00;
    COFLAM[14] = -1.31586198E-01;
    COFLAM[15] = 5.22416151E-03;
    COFLAM[16] = 9.09561056E+00;
    COFLAM[17] = -1.12146910E+00;
    COFLAM[18] = 2.39077996E-01;
    COFLAM[19] = -9.74338015E-03;
    COFLAM[20] = 1.81612053E+01;
    COFLAM[21] = -6.74137053E+00;
    COFLAM[22] = 1.21372119E+00;
    COFLAM[23] = -6.11027962E-02;
    COFLAM[24] = 3.32637961E+00;
    COFLAM[25] = 4.13455985E-01;
    COFLAM[26] = 1.13001798E-01;
    COFLAM[27] = -7.33441917E-03;
    COFLAM[28] = 2.91645348E+00;
    COFLAM[29] = 4.58703148E-01;
    COFLAM[30] = 1.38770970E-01;
    COFLAM[31] = -9.95034889E-03;
    COFLAM[32] = 1.42383309E+00;
    COFLAM[33] = 1.96214031E+00;
    COFLAM[34] = -1.79397210E-01;
    COFLAM[35] = 8.12672879E-03;
    COFLAM[36] = 1.36646194E+01;
    COFLAM[37] = -3.24578007E+00;
    COFLAM[38] = 5.66384436E-01;
    COFLAM[39] = -2.59027628E-02;
    COFLAM[40] = 8.22510964E+00;
    COFLAM[41] = -1.73468379E+00;
    COFLAM[42] = 4.32494779E-01;
    COFLAM[43] = -2.30931364E-02;
    COFLAM[44] = 8.17515440E+00;
    COFLAM[45] = -1.53836440E+00;
    COFLAM[46] = 3.68036945E-01;
    COFLAM[47] = -1.90917513E-02;
    COFLAM[48] = -8.74831432E+00;
    COFLAM[49] = 4.79275291E+00;
    COFLAM[50] = -4.18685061E-01;
    COFLAM[51] = 1.35210242E-02;
    COFLAM[52] = 1.06317650E+01;
    COFLAM[53] = -3.96201529E+00;
    COFLAM[54] = 8.65922550E-01;
    COFLAM[55] = -4.80178076E-02;
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
    COFD[12] = -1.22181183E+01;
    COFD[13] = 2.70415313E+00;
    COFD[14] = -1.41236971E-01;
    COFD[15] = 6.32236816E-03;
    COFD[16] = -1.12687251E+01;
    COFD[17] = 2.43094296E+00;
    COFD[18] = -1.03798673E-01;
    COFD[19] = 4.60962717E-03;
    COFD[20] = -1.73864044E+01;
    COFD[21] = 4.71143088E+00;
    COFD[22] = -3.95288626E-01;
    COFD[23] = 1.70702272E-02;
    COFD[24] = -1.22190241E+01;
    COFD[25] = 2.70415313E+00;
    COFD[26] = -1.41236971E-01;
    COFD[27] = 6.32236816E-03;
    COFD[28] = -1.22198776E+01;
    COFD[29] = 2.70415313E+00;
    COFD[30] = -1.41236971E-01;
    COFD[31] = 6.32236816E-03;
    COFD[32] = -1.12314500E+01;
    COFD[33] = 2.35359960E+00;
    COFD[34] = -9.33663289E-02;
    COFD[35] = 4.13977197E-03;
    COFD[36] = -1.12527514E+01;
    COFD[37] = 2.43094296E+00;
    COFD[38] = -1.03798673E-01;
    COFD[39] = 4.60962717E-03;
    COFD[40] = -1.30060041E+01;
    COFD[41] = 2.98419920E+00;
    COFD[42] = -1.78783571E-01;
    COFD[43] = 8.00253963E-03;
    COFD[44] = -1.20607690E+01;
    COFD[45] = 2.61969379E+00;
    COFD[46] = -1.29638429E-01;
    COFD[47] = 5.79050588E-03;
    COFD[48] = -1.43978662E+01;
    COFD[49] = 3.49721576E+00;
    COFD[50] = -2.45465191E-01;
    COFD[51] = 1.08948372E-02;
    COFD[52] = -1.64242337E+01;
    COFD[53] = 4.26219537E+00;
    COFD[54] = -3.41524254E-01;
    COFD[55] = 1.49232070E-02;
    COFD[56] = -1.19370371E+01;
    COFD[57] = 2.99054752E+00;
    COFD[58] = -1.79624448E-01;
    COFD[59] = 8.03970815E-03;
    COFD[60] = -1.51395399E+01;
    COFD[61] = 4.36621619E+00;
    COFD[62] = -3.53866950E-01;
    COFD[63] = 1.54097445E-02;
    COFD[64] = -1.39677477E+01;
    COFD[65] = 3.71279442E+00;
    COFD[66] = -2.72718508E-01;
    COFD[67] = 1.20446550E-02;
    COFD[68] = -1.51442279E+01;
    COFD[69] = 4.03719698E+00;
    COFD[70] = -3.13407940E-01;
    COFD[71] = 1.37483092E-02;
    COFD[72] = -1.39695071E+01;
    COFD[73] = 3.71279442E+00;
    COFD[74] = -2.72718508E-01;
    COFD[75] = 1.20446550E-02;
    COFD[76] = -1.80638064E+01;
    COFD[77] = 4.96477835E+00;
    COFD[78] = -3.98489680E-01;
    COFD[79] = 1.60121621E-02;
    COFD[80] = -1.51446944E+01;
    COFD[81] = 4.03719698E+00;
    COFD[82] = -3.13407940E-01;
    COFD[83] = 1.37483092E-02;
    COFD[84] = -1.51451337E+01;
    COFD[85] = 4.03719698E+00;
    COFD[86] = -3.13407940E-01;
    COFD[87] = 1.37483092E-02;
    COFD[88] = -1.38406189E+01;
    COFD[89] = 3.58824515E+00;
    COFD[90] = -2.56761889E-01;
    COFD[91] = 1.13618125E-02;
    COFD[92] = -1.39610098E+01;
    COFD[93] = 3.71279442E+00;
    COFD[94] = -2.72718508E-01;
    COFD[95] = 1.20446550E-02;
    COFD[96] = -1.61362164E+01;
    COFD[97] = 4.35660780E+00;
    COFD[98] = -3.52659055E-01;
    COFD[99] = 1.53589488E-02;
    COFD[100] = -1.49673213E+01;
    COFD[101] = 3.95033191E+00;
    COFD[102] = -3.02754076E-01;
    COFD[103] = 1.33124608E-02;
    COFD[104] = -1.76943558E+01;
    COFD[105] = 4.88672714E+00;
    COFD[106] = -4.14778255E-01;
    COFD[107] = 1.77823474E-02;
    COFD[108] = -1.85331662E+01;
    COFD[109] = 5.04618677E+00;
    COFD[110] = -4.14801122E-01;
    COFD[111] = 1.69487185E-02;
    COFD[112] = -1.12653981E+01;
    COFD[113] = 2.43094296E+00;
    COFD[114] = -1.03798673E-01;
    COFD[115] = 4.60962717E-03;
    COFD[116] = -1.39677477E+01;
    COFD[117] = 3.71279442E+00;
    COFD[118] = -2.72718508E-01;
    COFD[119] = 1.20446550E-02;
    COFD[120] = -1.37174845E+01;
    COFD[121] = 3.11889373E+00;
    COFD[122] = -1.96402933E-01;
    COFD[123] = 8.77180880E-03;
    COFD[124] = -1.49340210E+01;
    COFD[125] = 3.43509376E+00;
    COFD[126] = -2.37713783E-01;
    COFD[127] = 1.05726006E-02;
    COFD[128] = -1.37325251E+01;
    COFD[129] = 3.11889373E+00;
    COFD[130] = -1.96402933E-01;
    COFD[131] = 8.77180880E-03;
    COFD[132] = -1.90632945E+01;
    COFD[133] = 5.00590780E+00;
    COFD[134] = -4.24055652E-01;
    COFD[135] = 1.79326322E-02;
    COFD[136] = -1.49391368E+01;
    COFD[137] = 3.43509376E+00;
    COFD[138] = -2.37713783E-01;
    COFD[139] = 1.05726006E-02;
    COFD[140] = -1.49439976E+01;
    COFD[141] = 3.43509376E+00;
    COFD[142] = -2.37713783E-01;
    COFD[143] = 1.05726006E-02;
    COFD[144] = -1.35760163E+01;
    COFD[145] = 3.02294501E+00;
    COFD[146] = -1.83901334E-01;
    COFD[147] = 8.22811760E-03;
    COFD[148] = -1.36633004E+01;
    COFD[149] = 3.11889373E+00;
    COFD[150] = -1.96402933E-01;
    COFD[151] = 8.77180880E-03;
    COFD[152] = -1.56003535E+01;
    COFD[153] = 3.70484350E+00;
    COFD[154] = -2.71706480E-01;
    COFD[155] = 1.20016288E-02;
    COFD[156] = -1.47359425E+01;
    COFD[157] = 3.34699328E+00;
    COFD[158] = -2.26393424E-01;
    COFD[159] = 1.00872697E-02;
    COFD[160] = -1.74119025E+01;
    COFD[161] = 4.28601449E+00;
    COFD[162] = -3.44182880E-01;
    COFD[163] = 1.50201783E-02;
    COFD[164] = -1.92502106E+01;
    COFD[165] = 4.94108508E+00;
    COFD[166] = -4.19008197E-01;
    COFD[167] = 1.78499457E-02;
    COFD[168] = -1.22181183E+01;
    COFD[169] = 2.70415313E+00;
    COFD[170] = -1.41236971E-01;
    COFD[171] = 6.32236816E-03;
    COFD[172] = -1.51442279E+01;
    COFD[173] = 4.03719698E+00;
    COFD[174] = -3.13407940E-01;
    COFD[175] = 1.37483092E-02;
    COFD[176] = -1.49340210E+01;
    COFD[177] = 3.43509376E+00;
    COFD[178] = -2.37713783E-01;
    COFD[179] = 1.05726006E-02;
    COFD[180] = -1.60936570E+01;
    COFD[181] = 3.70633871E+00;
    COFD[182] = -2.71897253E-01;
    COFD[183] = 1.20097588E-02;
    COFD[184] = -1.49541774E+01;
    COFD[185] = 3.43509376E+00;
    COFD[186] = -2.37713783E-01;
    COFD[187] = 1.05726006E-02;
    COFD[188] = -1.99035583E+01;
    COFD[189] = 5.01694644E+00;
    COFD[190] = -4.08963011E-01;
    COFD[191] = 1.66143416E-02;
    COFD[192] = -1.61013505E+01;
    COFD[193] = 3.70633871E+00;
    COFD[194] = -2.71897253E-01;
    COFD[195] = 1.20097588E-02;
    COFD[196] = -1.61086977E+01;
    COFD[197] = 3.70633871E+00;
    COFD[198] = -2.71897253E-01;
    COFD[199] = 1.20097588E-02;
    COFD[200] = -1.47065699E+01;
    COFD[201] = 3.32403491E+00;
    COFD[202] = -2.23413853E-01;
    COFD[203] = 9.95822567E-03;
    COFD[204] = -1.48630063E+01;
    COFD[205] = 3.43509376E+00;
    COFD[206] = -2.37713783E-01;
    COFD[207] = 1.05726006E-02;
    COFD[208] = -1.68277030E+01;
    COFD[209] = 4.03054927E+00;
    COFD[210] = -3.12591852E-01;
    COFD[211] = 1.37148695E-02;
    COFD[212] = -1.58458281E+01;
    COFD[213] = 3.60600362E+00;
    COFD[214] = -2.59019961E-01;
    COFD[215] = 1.14576923E-02;
    COFD[216] = -1.87634092E+01;
    COFD[217] = 4.61060397E+00;
    COFD[218] = -3.83564503E-01;
    COFD[219] = 1.66168246E-02;
    COFD[220] = -2.01420005E+01;
    COFD[221] = 5.05721632E+00;
    COFD[222] = -4.26359426E-01;
    COFD[223] = 1.78564586E-02;
    COFD[224] = -1.12687251E+01;
    COFD[225] = 2.43094296E+00;
    COFD[226] = -1.03798673E-01;
    COFD[227] = 4.60962717E-03;
    COFD[228] = -1.39695071E+01;
    COFD[229] = 3.71279442E+00;
    COFD[230] = -2.72718508E-01;
    COFD[231] = 1.20446550E-02;
    COFD[232] = -1.37325251E+01;
    COFD[233] = 3.11889373E+00;
    COFD[234] = -1.96402933E-01;
    COFD[235] = 8.77180880E-03;
    COFD[236] = -1.49541774E+01;
    COFD[237] = 3.43509376E+00;
    COFD[238] = -2.37713783E-01;
    COFD[239] = 1.05726006E-02;
    COFD[240] = -1.37480322E+01;
    COFD[241] = 3.11889373E+00;
    COFD[242] = -1.96402933E-01;
    COFD[243] = 8.77180880E-03;
    COFD[244] = -1.90792409E+01;
    COFD[245] = 5.00590780E+00;
    COFD[246] = -4.24055652E-01;
    COFD[247] = 1.79326322E-02;
    COFD[248] = -1.49595048E+01;
    COFD[249] = 3.43509376E+00;
    COFD[250] = -2.37713783E-01;
    COFD[251] = 1.05726006E-02;
    COFD[252] = -1.49645688E+01;
    COFD[253] = 3.43509376E+00;
    COFD[254] = -2.37713783E-01;
    COFD[255] = 1.05726006E-02;
    COFD[256] = -1.35888876E+01;
    COFD[257] = 3.02294501E+00;
    COFD[258] = -1.83901334E-01;
    COFD[259] = 8.22811760E-03;
    COFD[260] = -1.36767754E+01;
    COFD[261] = 3.11889373E+00;
    COFD[262] = -1.96402933E-01;
    COFD[263] = 8.77180880E-03;
    COFD[264] = -1.56143922E+01;
    COFD[265] = 3.70484350E+00;
    COFD[266] = -2.71706480E-01;
    COFD[267] = 1.20016288E-02;
    COFD[268] = -1.47551678E+01;
    COFD[269] = 3.34699328E+00;
    COFD[270] = -2.26393424E-01;
    COFD[271] = 1.00872697E-02;
    COFD[272] = -1.74341216E+01;
    COFD[273] = 4.28601449E+00;
    COFD[274] = -3.44182880E-01;
    COFD[275] = 1.50201783E-02;
    COFD[276] = -1.92696867E+01;
    COFD[277] = 4.94108508E+00;
    COFD[278] = -4.19008197E-01;
    COFD[279] = 1.78499457E-02;
    COFD[280] = -1.73864044E+01;
    COFD[281] = 4.71143088E+00;
    COFD[282] = -3.95288626E-01;
    COFD[283] = 1.70702272E-02;
    COFD[284] = -1.80638064E+01;
    COFD[285] = 4.96477835E+00;
    COFD[286] = -3.98489680E-01;
    COFD[287] = 1.60121621E-02;
    COFD[288] = -1.90632945E+01;
    COFD[289] = 5.00590780E+00;
    COFD[290] = -4.24055652E-01;
    COFD[291] = 1.79326322E-02;
    COFD[292] = -1.99035583E+01;
    COFD[293] = 5.01694644E+00;
    COFD[294] = -4.08963011E-01;
    COFD[295] = 1.66143416E-02;
    COFD[296] = -1.90792409E+01;
    COFD[297] = 5.00590780E+00;
    COFD[298] = -4.24055652E-01;
    COFD[299] = 1.79326322E-02;
    COFD[300] = -1.16123849E+01;
    COFD[301] = 8.27754782E-01;
    COFD[302] = 2.52262233E-01;
    COFD[303] = -1.62567414E-02;
    COFD[304] = -1.98455231E+01;
    COFD[305] = 5.07004702E+00;
    COFD[306] = -4.23605778E-01;
    COFD[307] = 1.75592300E-02;
    COFD[308] = -1.98507823E+01;
    COFD[309] = 5.07004702E+00;
    COFD[310] = -4.23605778E-01;
    COFD[311] = 1.75592300E-02;
    COFD[312] = -1.89840016E+01;
    COFD[313] = 4.95746432E+00;
    COFD[314] = -4.20580224E-01;
    COFD[315] = 1.78956385E-02;
    COFD[316] = -1.90060763E+01;
    COFD[317] = 5.00590780E+00;
    COFD[318] = -4.24055652E-01;
    COFD[319] = 1.79326322E-02;
    COFD[320] = -1.97582846E+01;
    COFD[321] = 4.96750408E+00;
    COFD[322] = -3.99127523E-01;
    COFD[323] = 1.60511657E-02;
    COFD[324] = -1.99647405E+01;
    COFD[325] = 5.05179386E+00;
    COFD[326] = -4.16351103E-01;
    COFD[327] = 1.70488551E-02;
    COFD[328] = -1.82187624E+01;
    COFD[329] = 3.93854160E+00;
    COFD[330] = -2.28424632E-01;
    COFD[331] = 7.18603342E-03;
    COFD[332] = -1.48220109E+01;
    COFD[333] = 2.35400310E+00;
    COFD[334] = 1.20736855E-02;
    COFD[335] = -4.57625832E-03;
    COFD[336] = -1.22190241E+01;
    COFD[337] = 2.70415313E+00;
    COFD[338] = -1.41236971E-01;
    COFD[339] = 6.32236816E-03;
    COFD[340] = -1.51446944E+01;
    COFD[341] = 4.03719698E+00;
    COFD[342] = -3.13407940E-01;
    COFD[343] = 1.37483092E-02;
    COFD[344] = -1.49391368E+01;
    COFD[345] = 3.43509376E+00;
    COFD[346] = -2.37713783E-01;
    COFD[347] = 1.05726006E-02;
    COFD[348] = -1.61013505E+01;
    COFD[349] = 3.70633871E+00;
    COFD[350] = -2.71897253E-01;
    COFD[351] = 1.20097588E-02;
    COFD[352] = -1.49595048E+01;
    COFD[353] = 3.43509376E+00;
    COFD[354] = -2.37713783E-01;
    COFD[355] = 1.05726006E-02;
    COFD[356] = -1.98455231E+01;
    COFD[357] = 5.07004702E+00;
    COFD[358] = -4.23605778E-01;
    COFD[359] = 1.75592300E-02;
    COFD[360] = -1.61091642E+01;
    COFD[361] = 3.70633871E+00;
    COFD[362] = -2.71897253E-01;
    COFD[363] = 1.20097588E-02;
    COFD[364] = -1.61166279E+01;
    COFD[365] = 3.70633871E+00;
    COFD[366] = -2.71897253E-01;
    COFD[367] = 1.20097588E-02;
    COFD[368] = -1.47107546E+01;
    COFD[369] = 3.32403491E+00;
    COFD[370] = -2.23413853E-01;
    COFD[371] = 9.95822567E-03;
    COFD[372] = -1.48674418E+01;
    COFD[373] = 3.43509376E+00;
    COFD[374] = -2.37713783E-01;
    COFD[375] = 1.05726006E-02;
    COFD[376] = -1.68323783E+01;
    COFD[377] = 4.03054927E+00;
    COFD[378] = -3.12591852E-01;
    COFD[379] = 1.37148695E-02;
    COFD[380] = -1.58530066E+01;
    COFD[381] = 3.60600362E+00;
    COFD[382] = -2.59019961E-01;
    COFD[383] = 1.14576923E-02;
    COFD[384] = -1.87723293E+01;
    COFD[385] = 4.61060397E+00;
    COFD[386] = -3.83564503E-01;
    COFD[387] = 1.66168246E-02;
    COFD[388] = -2.01493154E+01;
    COFD[389] = 5.05721632E+00;
    COFD[390] = -4.26359426E-01;
    COFD[391] = 1.78564586E-02;
    COFD[392] = -1.22198776E+01;
    COFD[393] = 2.70415313E+00;
    COFD[394] = -1.41236971E-01;
    COFD[395] = 6.32236816E-03;
    COFD[396] = -1.51451337E+01;
    COFD[397] = 4.03719698E+00;
    COFD[398] = -3.13407940E-01;
    COFD[399] = 1.37483092E-02;
    COFD[400] = -1.49439976E+01;
    COFD[401] = 3.43509376E+00;
    COFD[402] = -2.37713783E-01;
    COFD[403] = 1.05726006E-02;
    COFD[404] = -1.61086977E+01;
    COFD[405] = 3.70633871E+00;
    COFD[406] = -2.71897253E-01;
    COFD[407] = 1.20097588E-02;
    COFD[408] = -1.49645688E+01;
    COFD[409] = 3.43509376E+00;
    COFD[410] = -2.37713783E-01;
    COFD[411] = 1.05726006E-02;
    COFD[412] = -1.98507823E+01;
    COFD[413] = 5.07004702E+00;
    COFD[414] = -4.23605778E-01;
    COFD[415] = 1.75592300E-02;
    COFD[416] = -1.61166279E+01;
    COFD[417] = 3.70633871E+00;
    COFD[418] = -2.71897253E-01;
    COFD[419] = 1.20097588E-02;
    COFD[420] = -1.61242048E+01;
    COFD[421] = 3.70633871E+00;
    COFD[422] = -2.71897253E-01;
    COFD[423] = 1.20097588E-02;
    COFD[424] = -1.47147236E+01;
    COFD[425] = 3.32403491E+00;
    COFD[426] = -2.23413853E-01;
    COFD[427] = 9.95822567E-03;
    COFD[428] = -1.48716505E+01;
    COFD[429] = 3.43509376E+00;
    COFD[430] = -2.37713783E-01;
    COFD[431] = 1.05726006E-02;
    COFD[432] = -1.68368168E+01;
    COFD[433] = 4.03054927E+00;
    COFD[434] = -3.12591852E-01;
    COFD[435] = 1.37148695E-02;
    COFD[436] = -1.58598550E+01;
    COFD[437] = 3.60600362E+00;
    COFD[438] = -2.59019961E-01;
    COFD[439] = 1.14576923E-02;
    COFD[440] = -1.87808686E+01;
    COFD[441] = 4.61060397E+00;
    COFD[442] = -3.83564503E-01;
    COFD[443] = 1.66168246E-02;
    COFD[444] = -2.01562959E+01;
    COFD[445] = 5.05721632E+00;
    COFD[446] = -4.26359426E-01;
    COFD[447] = 1.78564586E-02;
    COFD[448] = -1.12314500E+01;
    COFD[449] = 2.35359960E+00;
    COFD[450] = -9.33663289E-02;
    COFD[451] = 4.13977197E-03;
    COFD[452] = -1.38406189E+01;
    COFD[453] = 3.58824515E+00;
    COFD[454] = -2.56761889E-01;
    COFD[455] = 1.13618125E-02;
    COFD[456] = -1.35760163E+01;
    COFD[457] = 3.02294501E+00;
    COFD[458] = -1.83901334E-01;
    COFD[459] = 8.22811760E-03;
    COFD[460] = -1.47065699E+01;
    COFD[461] = 3.32403491E+00;
    COFD[462] = -2.23413853E-01;
    COFD[463] = 9.95822567E-03;
    COFD[464] = -1.35888876E+01;
    COFD[465] = 3.02294501E+00;
    COFD[466] = -1.83901334E-01;
    COFD[467] = 8.22811760E-03;
    COFD[468] = -1.89840016E+01;
    COFD[469] = 4.95746432E+00;
    COFD[470] = -4.20580224E-01;
    COFD[471] = 1.78956385E-02;
    COFD[472] = -1.47107546E+01;
    COFD[473] = 3.32403491E+00;
    COFD[474] = -2.23413853E-01;
    COFD[475] = 9.95822567E-03;
    COFD[476] = -1.47147236E+01;
    COFD[477] = 3.32403491E+00;
    COFD[478] = -2.23413853E-01;
    COFD[479] = 9.95822567E-03;
    COFD[480] = -1.34113597E+01;
    COFD[481] = 2.91879827E+00;
    COFD[482] = -1.70127286E-01;
    COFD[483] = 7.62035353E-03;
    COFD[484] = -1.35291979E+01;
    COFD[485] = 3.02294501E+00;
    COFD[486] = -1.83901334E-01;
    COFD[487] = 8.22811760E-03;
    COFD[488] = -1.53581150E+01;
    COFD[489] = 3.58137147E+00;
    COFD[490] = -2.55890214E-01;
    COFD[491] = 1.13249037E-02;
    COFD[492] = -1.44936367E+01;
    COFD[493] = 3.22906684E+00;
    COFD[494] = -2.11023286E-01;
    COFD[495] = 9.41886188E-03;
    COFD[496] = -1.71713321E+01;
    COFD[497] = 4.18577731E+00;
    COFD[498] = -3.32464208E-01;
    COFD[499] = 1.45662717E-02;
    COFD[500] = -1.91185434E+01;
    COFD[501] = 4.88857998E+00;
    COFD[502] = -4.14895102E-01;
    COFD[503] = 1.77823222E-02;
    COFD[504] = -1.12527514E+01;
    COFD[505] = 2.43094296E+00;
    COFD[506] = -1.03798673E-01;
    COFD[507] = 4.60962717E-03;
    COFD[508] = -1.39610098E+01;
    COFD[509] = 3.71279442E+00;
    COFD[510] = -2.72718508E-01;
    COFD[511] = 1.20446550E-02;
    COFD[512] = -1.36633004E+01;
    COFD[513] = 3.11889373E+00;
    COFD[514] = -1.96402933E-01;
    COFD[515] = 8.77180880E-03;
    COFD[516] = -1.48630063E+01;
    COFD[517] = 3.43509376E+00;
    COFD[518] = -2.37713783E-01;
    COFD[519] = 1.05726006E-02;
    COFD[520] = -1.36767754E+01;
    COFD[521] = 3.11889373E+00;
    COFD[522] = -1.96402933E-01;
    COFD[523] = 8.77180880E-03;
    COFD[524] = -1.90060763E+01;
    COFD[525] = 5.00590780E+00;
    COFD[526] = -4.24055652E-01;
    COFD[527] = 1.79326322E-02;
    COFD[528] = -1.48674418E+01;
    COFD[529] = 3.43509376E+00;
    COFD[530] = -2.37713783E-01;
    COFD[531] = 1.05726006E-02;
    COFD[532] = -1.48716505E+01;
    COFD[533] = 3.43509376E+00;
    COFD[534] = -2.37713783E-01;
    COFD[535] = 1.05726006E-02;
    COFD[536] = -1.35291979E+01;
    COFD[537] = 3.02294501E+00;
    COFD[538] = -1.83901334E-01;
    COFD[539] = 8.22811760E-03;
    COFD[540] = -1.36144184E+01;
    COFD[541] = 3.11889373E+00;
    COFD[542] = -1.96402933E-01;
    COFD[543] = 8.77180880E-03;
    COFD[544] = -1.55495540E+01;
    COFD[545] = 3.70484350E+00;
    COFD[546] = -2.71706480E-01;
    COFD[547] = 1.20016288E-02;
    COFD[548] = -1.46679365E+01;
    COFD[549] = 3.34699328E+00;
    COFD[550] = -2.26393424E-01;
    COFD[551] = 1.00872697E-02;
    COFD[552] = -1.73343060E+01;
    COFD[553] = 4.28601449E+00;
    COFD[554] = -3.44182880E-01;
    COFD[555] = 1.50201783E-02;
    COFD[556] = -1.91813922E+01;
    COFD[557] = 4.94108508E+00;
    COFD[558] = -4.19008197E-01;
    COFD[559] = 1.78499457E-02;
    COFD[560] = -1.30060041E+01;
    COFD[561] = 2.98419920E+00;
    COFD[562] = -1.78783571E-01;
    COFD[563] = 8.00253963E-03;
    COFD[564] = -1.61362164E+01;
    COFD[565] = 4.35660780E+00;
    COFD[566] = -3.52659055E-01;
    COFD[567] = 1.53589488E-02;
    COFD[568] = -1.56003535E+01;
    COFD[569] = 3.70484350E+00;
    COFD[570] = -2.71706480E-01;
    COFD[571] = 1.20016288E-02;
    COFD[572] = -1.68277030E+01;
    COFD[573] = 4.03054927E+00;
    COFD[574] = -3.12591852E-01;
    COFD[575] = 1.37148695E-02;
    COFD[576] = -1.56143922E+01;
    COFD[577] = 3.70484350E+00;
    COFD[578] = -2.71706480E-01;
    COFD[579] = 1.20016288E-02;
    COFD[580] = -1.97582846E+01;
    COFD[581] = 4.96750408E+00;
    COFD[582] = -3.99127523E-01;
    COFD[583] = 1.60511657E-02;
    COFD[584] = -1.68323783E+01;
    COFD[585] = 4.03054927E+00;
    COFD[586] = -3.12591852E-01;
    COFD[587] = 1.37148695E-02;
    COFD[588] = -1.68368168E+01;
    COFD[589] = 4.03054927E+00;
    COFD[590] = -3.12591852E-01;
    COFD[591] = 1.37148695E-02;
    COFD[592] = -1.53581150E+01;
    COFD[593] = 3.58137147E+00;
    COFD[594] = -2.55890214E-01;
    COFD[595] = 1.13249037E-02;
    COFD[596] = -1.55495540E+01;
    COFD[597] = 3.70484350E+00;
    COFD[598] = -2.71706480E-01;
    COFD[599] = 1.20016288E-02;
    COFD[600] = -1.76381609E+01;
    COFD[601] = 4.34699543E+00;
    COFD[602] = -3.51450562E-01;
    COFD[603] = 1.53081221E-02;
    COFD[604] = -1.66149558E+01;
    COFD[605] = 3.94349539E+00;
    COFD[606] = -3.01913683E-01;
    COFD[607] = 1.32780377E-02;
    COFD[608] = -1.93972356E+01;
    COFD[609] = 4.88315812E+00;
    COFD[610] = -4.14471408E-01;
    COFD[611] = 1.77754868E-02;
    COFD[612] = -2.02092974E+01;
    COFD[613] = 5.04826260E+00;
    COFD[614] = -4.15332602E-01;
    COFD[615] = 1.69822815E-02;
    COFD[616] = -1.20607690E+01;
    COFD[617] = 2.61969379E+00;
    COFD[618] = -1.29638429E-01;
    COFD[619] = 5.79050588E-03;
    COFD[620] = -1.49673213E+01;
    COFD[621] = 3.95033191E+00;
    COFD[622] = -3.02754076E-01;
    COFD[623] = 1.33124608E-02;
    COFD[624] = -1.47359425E+01;
    COFD[625] = 3.34699328E+00;
    COFD[626] = -2.26393424E-01;
    COFD[627] = 1.00872697E-02;
    COFD[628] = -1.58458281E+01;
    COFD[629] = 3.60600362E+00;
    COFD[630] = -2.59019961E-01;
    COFD[631] = 1.14576923E-02;
    COFD[632] = -1.47551678E+01;
    COFD[633] = 3.34699328E+00;
    COFD[634] = -2.26393424E-01;
    COFD[635] = 1.00872697E-02;
    COFD[636] = -1.99647405E+01;
    COFD[637] = 5.05179386E+00;
    COFD[638] = -4.16351103E-01;
    COFD[639] = 1.70488551E-02;
    COFD[640] = -1.58530066E+01;
    COFD[641] = 3.60600362E+00;
    COFD[642] = -2.59019961E-01;
    COFD[643] = 1.14576923E-02;
    COFD[644] = -1.58598550E+01;
    COFD[645] = 3.60600362E+00;
    COFD[646] = -2.59019961E-01;
    COFD[647] = 1.14576923E-02;
    COFD[648] = -1.44936367E+01;
    COFD[649] = 3.22906684E+00;
    COFD[650] = -2.11023286E-01;
    COFD[651] = 9.41886188E-03;
    COFD[652] = -1.46679365E+01;
    COFD[653] = 3.34699328E+00;
    COFD[654] = -2.26393424E-01;
    COFD[655] = 1.00872697E-02;
    COFD[656] = -1.66149558E+01;
    COFD[657] = 3.94349539E+00;
    COFD[658] = -3.01913683E-01;
    COFD[659] = 1.32780377E-02;
    COFD[660] = -1.56423580E+01;
    COFD[661] = 3.52412711E+00;
    COFD[662] = -2.48745351E-01;
    COFD[663] = 1.10277551E-02;
    COFD[664] = -1.85324360E+01;
    COFD[665] = 4.52748688E+00;
    COFD[666] = -3.73847542E-01;
    COFD[667] = 1.62384117E-02;
    COFD[668] = -2.00026412E+01;
    COFD[669] = 5.01818529E+00;
    COFD[670] = -4.23776772E-01;
    COFD[671] = 1.78445623E-02;
    COFD[672] = -1.43978662E+01;
    COFD[673] = 3.49721576E+00;
    COFD[674] = -2.45465191E-01;
    COFD[675] = 1.08948372E-02;
    COFD[676] = -1.76943558E+01;
    COFD[677] = 4.88672714E+00;
    COFD[678] = -4.14778255E-01;
    COFD[679] = 1.77823474E-02;
    COFD[680] = -1.74119025E+01;
    COFD[681] = 4.28601449E+00;
    COFD[682] = -3.44182880E-01;
    COFD[683] = 1.50201783E-02;
    COFD[684] = -1.87634092E+01;
    COFD[685] = 4.61060397E+00;
    COFD[686] = -3.83564503E-01;
    COFD[687] = 1.66168246E-02;
    COFD[688] = -1.74341216E+01;
    COFD[689] = 4.28601449E+00;
    COFD[690] = -3.44182880E-01;
    COFD[691] = 1.50201783E-02;
    COFD[692] = -1.82187624E+01;
    COFD[693] = 3.93854160E+00;
    COFD[694] = -2.28424632E-01;
    COFD[695] = 7.18603342E-03;
    COFD[696] = -1.87723293E+01;
    COFD[697] = 4.61060397E+00;
    COFD[698] = -3.83564503E-01;
    COFD[699] = 1.66168246E-02;
    COFD[700] = -1.87808686E+01;
    COFD[701] = 4.61060397E+00;
    COFD[702] = -3.83564503E-01;
    COFD[703] = 1.66168246E-02;
    COFD[704] = -1.71713321E+01;
    COFD[705] = 4.18577731E+00;
    COFD[706] = -3.32464208E-01;
    COFD[707] = 1.45662717E-02;
    COFD[708] = -1.73343060E+01;
    COFD[709] = 4.28601449E+00;
    COFD[710] = -3.44182880E-01;
    COFD[711] = 1.50201783E-02;
    COFD[712] = -1.93972356E+01;
    COFD[713] = 4.88315812E+00;
    COFD[714] = -4.14471408E-01;
    COFD[715] = 1.77754868E-02;
    COFD[716] = -1.85324360E+01;
    COFD[717] = 4.52748688E+00;
    COFD[718] = -3.73847542E-01;
    COFD[719] = 1.62384117E-02;
    COFD[720] = -2.05810669E+01;
    COFD[721] = 5.07469434E+00;
    COFD[722] = -4.25340301E-01;
    COFD[723] = 1.76800795E-02;
    COFD[724] = -1.98561573E+01;
    COFD[725] = 4.57685026E+00;
    COFD[726] = -3.30016794E-01;
    COFD[727] = 1.23264865E-02;
    COFD[728] = -1.64242337E+01;
    COFD[729] = 4.26219537E+00;
    COFD[730] = -3.41524254E-01;
    COFD[731] = 1.49232070E-02;
    COFD[732] = -1.85331662E+01;
    COFD[733] = 5.04618677E+00;
    COFD[734] = -4.14801122E-01;
    COFD[735] = 1.69487185E-02;
    COFD[736] = -1.92502106E+01;
    COFD[737] = 4.94108508E+00;
    COFD[738] = -4.19008197E-01;
    COFD[739] = 1.78499457E-02;
    COFD[740] = -2.01420005E+01;
    COFD[741] = 5.05721632E+00;
    COFD[742] = -4.26359426E-01;
    COFD[743] = 1.78564586E-02;
    COFD[744] = -1.92696867E+01;
    COFD[745] = 4.94108508E+00;
    COFD[746] = -4.19008197E-01;
    COFD[747] = 1.78499457E-02;
    COFD[748] = -1.48220109E+01;
    COFD[749] = 2.35400310E+00;
    COFD[750] = 1.20736855E-02;
    COFD[751] = -4.57625832E-03;
    COFD[752] = -2.01493154E+01;
    COFD[753] = 5.05721632E+00;
    COFD[754] = -4.26359426E-01;
    COFD[755] = 1.78564586E-02;
    COFD[756] = -2.01562959E+01;
    COFD[757] = 5.05721632E+00;
    COFD[758] = -4.26359426E-01;
    COFD[759] = 1.78564586E-02;
    COFD[760] = -1.91185434E+01;
    COFD[761] = 4.88857998E+00;
    COFD[762] = -4.14895102E-01;
    COFD[763] = 1.77823222E-02;
    COFD[764] = -1.91813922E+01;
    COFD[765] = 4.94108508E+00;
    COFD[766] = -4.19008197E-01;
    COFD[767] = 1.78499457E-02;
    COFD[768] = -2.02092974E+01;
    COFD[769] = 5.04826260E+00;
    COFD[770] = -4.15332602E-01;
    COFD[771] = 1.69822815E-02;
    COFD[772] = -2.00026412E+01;
    COFD[773] = 5.01818529E+00;
    COFD[774] = -4.23776772E-01;
    COFD[775] = 1.78445623E-02;
    COFD[776] = -1.98561573E+01;
    COFD[777] = 4.57685026E+00;
    COFD[778] = -3.30016794E-01;
    COFD[779] = 1.23264865E-02;
    COFD[780] = -1.61938051E+01;
    COFD[781] = 2.80507926E+00;
    COFD[782] = -5.55394339E-02;
    COFD[783] = -1.30364179E-03;
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
    COFTD[12] = 3.81864172E-01;
    COFTD[13] = 1.84117353E-04;
    COFTD[14] = -9.79617476E-08;
    COFTD[15] = 1.62542227E-11;
    COFTD[16] = 3.75475346E-01;
    COFTD[17] = 9.73577933E-05;
    COFTD[18] = -4.94393493E-08;
    COFTD[19] = 8.33305051E-12;
    COFTD[20] = 1.95123509E-03;
    COFTD[21] = 6.69470998E-04;
    COFTD[22] = -3.12148757E-07;
    COFTD[23] = 4.52949938E-11;
    COFTD[24] = 3.83342059E-01;
    COFTD[25] = 1.84829923E-04;
    COFTD[26] = -9.83408783E-08;
    COFTD[27] = 1.63171297E-11;
    COFTD[28] = 3.84737257E-01;
    COFTD[29] = 1.85502623E-04;
    COFTD[30] = -9.86987963E-08;
    COFTD[31] = 1.63765169E-11;
    COFTD[32] = 3.49591226E-01;
    COFTD[33] = 6.73643711E-05;
    COFTD[34] = -3.23868084E-08;
    COFTD[35] = 5.44525031E-12;
    COFTD[36] = 3.48688402E-01;
    COFTD[37] = 9.04121502E-05;
    COFTD[38] = -4.59122760E-08;
    COFTD[39] = 7.73855886E-12;
    COFTD[40] = 2.85595384E-01;
    COFTD[41] = 2.28510573E-04;
    COFTD[42] = -1.21626586E-07;
    COFTD[43] = 1.97428169E-11;
    COFTD[44] = 3.87405318E-01;
    COFTD[45] = 1.56883797E-04;
    COFTD[46] = -8.29309791E-08;
    COFTD[47] = 1.38460299E-11;
    COFTD[48] = 2.47129011E-01;
    COFTD[49] = 4.49395677E-04;
    COFTD[50] = -2.32030740E-07;
    COFTD[51] = 3.62578797E-11;
    COFTD[52] = 8.88011309E-02;
    COFTD[53] = 6.35611803E-04;
    COFTD[54] = -3.09074957E-07;
    COFTD[55] = 4.61052654E-11;
    COFTD[56] = 1.26715692E-01;
    COFTD[57] = 1.02530485E-04;
    COFTD[58] = -5.45604892E-08;
    COFTD[59] = 8.85181063E-12;
    COFTD[60] = 0.00000000E+00;
    COFTD[61] = 0.00000000E+00;
    COFTD[62] = 0.00000000E+00;
    COFTD[63] = 0.00000000E+00;
    COFTD[64] = 1.93107545E-01;
    COFTD[65] = 5.04123759E-04;
    COFTD[66] = -2.55796068E-07;
    COFTD[67] = 3.93758307E-11;
    COFTD[68] = 1.39745676E-01;
    COFTD[69] = 6.29810814E-04;
    COFTD[70] = -3.11694011E-07;
    COFTD[71] = 4.70755830E-11;
    COFTD[72] = 1.94560455E-01;
    COFTD[73] = 5.07916706E-04;
    COFTD[74] = -2.57720638E-07;
    COFTD[75] = 3.96720883E-11;
    COFTD[76] = -1.60928523E-01;
    COFTD[77] = 8.01685562E-04;
    COFTD[78] = -3.24976618E-07;
    COFTD[79] = 4.31958164E-11;
    COFTD[80] = 1.40015055E-01;
    COFTD[81] = 6.31024860E-04;
    COFTD[82] = -3.12294843E-07;
    COFTD[83] = 4.71663275E-11;
    COFTD[84] = 1.40268928E-01;
    COFTD[85] = 6.32169024E-04;
    COFTD[86] = -3.12861091E-07;
    COFTD[87] = 4.72518487E-11;
    COFTD[88] = 2.07584420E-01;
    COFTD[89] = 4.49536077E-04;
    COFTD[90] = -2.30168774E-07;
    COFTD[91] = 3.56994231E-11;
    COFTD[92] = 1.87590287E-01;
    COFTD[93] = 4.89720486E-04;
    COFTD[94] = -2.48487743E-07;
    COFTD[95] = 3.82508275E-11;
    COFTD[96] = 6.80931000E-02;
    COFTD[97] = 6.56715864E-04;
    COFTD[98] = -3.16406689E-07;
    COFTD[99] = 4.69046855E-11;
    COFTD[100] = 1.58726794E-01;
    COFTD[101] = 5.96753380E-04;
    COFTD[102] = -2.97673771E-07;
    COFTD[103] = 4.52193864E-11;
    COFTD[104] = -3.84966170E-02;
    COFTD[105] = 8.34794926E-04;
    COFTD[106] = -3.81031231E-07;
    COFTD[107] = 5.45531902E-11;
    COFTD[108] = -1.52598674E-01;
    COFTD[109] = 8.48164645E-04;
    COFTD[110] = -3.52118423E-07;
    COFTD[111] = 4.75234393E-11;
}

/* Replace this routine with the one generated by the Gauss Jordan solver of DW */
AMREX_GPU_HOST_DEVICE void sgjsolve(double* A, double* x, double* b) {
    amrex::Abort("sgjsolve not implemented, choose a different solver ");
}

/* Replace this routine with the one generated by the Gauss Jordan solver of DW */
AMREX_GPU_HOST_DEVICE void sgjsolve_simplified(double* A, double* x, double* b) {
    amrex::Abort("sgjsolve_simplified not implemented, choose a different solver ");
}

