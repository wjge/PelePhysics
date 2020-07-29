#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <vector>
#include <AMReX_Gpu.H>

extern "C"
{
AMREX_GPU_HOST_DEVICE void get_imw(double imw_new[]);
AMREX_GPU_HOST_DEVICE void get_mw(double mw_new[]);
void atomicWeight(double *  awt);
void molecularWeight(double *  wt);
AMREX_GPU_HOST_DEVICE void gibbs(double *  species, double *  tc);
AMREX_GPU_HOST_DEVICE void gibbs_qss(double *  species, double *  tc);
AMREX_GPU_HOST_DEVICE void helmholtz(double *  species, double *  tc);
AMREX_GPU_HOST_DEVICE void speciesInternalEnergy(double *  species, double *  tc);
AMREX_GPU_HOST_DEVICE void speciesEnthalpy(double *  species, double *  tc);
AMREX_GPU_HOST_DEVICE void speciesEntropy(double *  species, double *  tc);
AMREX_GPU_HOST_DEVICE void cp_R(double *  species, double *  tc);
AMREX_GPU_HOST_DEVICE void cv_R(double *  species, double *  tc);
void equilibriumConstants(double *  kc, double *  g_RT, double T);
AMREX_GPU_HOST_DEVICE void productionRate(double *  wdot, double *  sc, double T);
AMREX_GPU_HOST_DEVICE void comp_qfqr(double *  q_f, double *  q_r, double *  sc, double *  qss_sc, double *  tc, double invT);
void comp_qss_sc(double * sc, double * sc_qss, double * tc, double invT);
void comp_qss_coeff(double *  qf_co, double *  qr_co, double *  sc, double *  tc, double invT);
#ifndef AMREX_USE_CUDA
void comp_k_f(double *  tc, double invT, double *  k_f);
void comp_Kc(double *  tc, double invT, double *  Kc);
#endif
AMREX_GPU_HOST_DEVICE void CKINIT();
AMREX_GPU_HOST_DEVICE void CKFINALIZE();
void CKINDX(int * mm, int * kk, int * ii, int * nfit );
void CKSYME_STR(amrex::Vector<std::string>& ename);
void CKSYME(int * kname, int * lenkname);
void CKSYMS_STR(amrex::Vector<std::string>& kname);
void CKSYMS(int * kname, int * lenkname);
void CKRP(double *  ru, double *  ruc, double *  pa);
void CKPX(double *  rho, double *  T, double *  x, double *  P);
AMREX_GPU_HOST_DEVICE void CKPY(double *  rho, double *  T, double *  y, double *  P);
void CKPC(double *  rho, double *  T, double *  c, double *  P);
void CKRHOX(double *  P, double *  T, double *  x, double *  rho);
AMREX_GPU_HOST_DEVICE void CKRHOY(double *  P, double *  T, double *  y, double *  rho);
void CKRHOC(double *  P, double *  T, double *  c, double *  rho);
void CKWT(double *  wt);
void CKAWT(double *  awt);
AMREX_GPU_HOST_DEVICE void CKMMWY(double *  y, double *  wtm);
void CKMMWX(double *  x, double *  wtm);
void CKMMWC(double *  c, double *  wtm);
AMREX_GPU_HOST_DEVICE void CKYTX(double *  y, double *  x);
void CKYTCP(double *  P, double *  T, double *  y, double *  c);
AMREX_GPU_HOST_DEVICE void CKYTCR(double *  rho, double *  T, double *  y, double *  c);
AMREX_GPU_HOST_DEVICE void CKXTY(double *  x, double *  y);
void CKXTCP(double *  P, double *  T, double *  x, double *  c);
void CKXTCR(double *  rho, double *  T, double *  x, double *  c);
void CKCTX(double *  c, double *  x);
void CKCTY(double *  c, double *  y);
void CKCPOR(double *  T, double *  cpor);
void CKHORT(double *  T, double *  hort);
void CKSOR(double *  T, double *  sor);
AMREX_GPU_HOST_DEVICE void CKCVMS(double *  T, double *  cvms);
AMREX_GPU_HOST_DEVICE void CKCPMS(double *  T, double *  cvms);
AMREX_GPU_HOST_DEVICE void CKUMS(double *  T, double *  ums);
AMREX_GPU_HOST_DEVICE void CKHMS(double *  T, double *  ums);
void CKGMS(double *  T, double *  gms);
void CKAMS(double *  T, double *  ams);
void CKSMS(double *  T, double *  sms);
void CKCPBL(double *  T, double *  x, double *  cpbl);
AMREX_GPU_HOST_DEVICE void CKCPBS(double *  T, double *  y, double *  cpbs);
void CKCVBL(double *  T, double *  x, double *  cpbl);
AMREX_GPU_HOST_DEVICE void CKCVBS(double *  T, double *  y, double *  cpbs);
void CKHBML(double *  T, double *  x, double *  hbml);
AMREX_GPU_HOST_DEVICE void CKHBMS(double *  T, double *  y, double *  hbms);
void CKUBML(double *  T, double *  x, double *  ubml);
AMREX_GPU_HOST_DEVICE void CKUBMS(double *  T, double *  y, double *  ubms);
void CKSBML(double *  P, double *  T, double *  x, double *  sbml);
void CKSBMS(double *  P, double *  T, double *  y, double *  sbms);
void CKGBML(double *  P, double *  T, double *  x, double *  gbml);
void CKGBMS(double *  P, double *  T, double *  y, double *  gbms);
void CKABML(double *  P, double *  T, double *  x, double *  abml);
void CKABMS(double *  P, double *  T, double *  y, double *  abms);
AMREX_GPU_HOST_DEVICE void CKWC(double *  T, double *  C, double *  wdot);
void CKWYP(double *  P, double *  T, double *  y, double *  wdot);
void CKWXP(double *  P, double *  T, double *  x, double *  wdot);
AMREX_GPU_HOST_DEVICE void CKWYR(double *  rho, double *  T, double *  y, double *  wdot);
void CKWXR(double *  rho, double *  T, double *  x, double *  wdot);
void CKNCF(int * ncf);
AMREX_GPU_HOST_DEVICE void DWDOT(double *  J, double *  sc, double *  T, int * consP);
AMREX_GPU_HOST_DEVICE void DWDOT_SIMPLIFIED(double *  J, double *  sc, double *  Tp, int * HP);
AMREX_GPU_HOST_DEVICE void SPARSITY_INFO(int * nJdata, int * consP, int NCELLS);
AMREX_GPU_HOST_DEVICE void SPARSITY_INFO_SYST(int * nJdata, int * consP, int NCELLS);
AMREX_GPU_HOST_DEVICE void SPARSITY_INFO_SYST_SIMPLIFIED(int * nJdata, int * consP);
AMREX_GPU_HOST_DEVICE void SPARSITY_PREPROC_CSC(int * rowVals, int * colPtrs, int * consP, int NCELLS);
AMREX_GPU_HOST_DEVICE void SPARSITY_PREPROC_CSR(int * colVals, int * rowPtrs, int * consP, int NCELLS, int base);
AMREX_GPU_HOST_DEVICE void SPARSITY_PREPROC_SYST_CSR(int * colVals, int * rowPtrs, int * consP, int NCELLS, int base);
AMREX_GPU_HOST_DEVICE void SPARSITY_PREPROC_SYST_SIMPLIFIED_CSC(int * rowVals, int * colPtrs, int * indx, int * consP);
AMREX_GPU_HOST_DEVICE void SPARSITY_PREPROC_SYST_SIMPLIFIED_CSR(int * colVals, int * rowPtr, int * consP, int base);
AMREX_GPU_HOST_DEVICE void aJacobian(double *  J, double *  sc, double T, int consP);
AMREX_GPU_HOST_DEVICE void aJacobian_precond(double *  J, double *  sc, double T, int HP);
AMREX_GPU_HOST_DEVICE void dcvpRdT(double *  species, double *  tc);
AMREX_GPU_HOST_DEVICE void GET_T_GIVEN_EY(double *  e, double *  y, double *  t, int *ierr);
AMREX_GPU_HOST_DEVICE void GET_T_GIVEN_HY(double *  h, double *  y, double *  t, int *ierr);
/*Transport function declarations */
void egtransetLENIMC(int* LENIMC);
void egtransetLENRMC(int* LENRMC);
void egtransetNO(int* NO);
void egtransetKK(int* KK);
void egtransetNLITE(int* NLITE);
void egtransetPATM(double* PATM);
void egtransetWT(double* WT);
void egtransetEPS(double* EPS);
void egtransetSIG(double* SIG);
void egtransetDIP(double* DIP);
void egtransetPOL(double* POL);
void egtransetZROT(double* ZROT);
void egtransetNLIN(int* NLIN);
void egtransetCOFETA(double* COFETA);
void egtransetCOFLAM(double* COFLAM);
void egtransetCOFD(double* COFD);
void egtransetKTDIF(int* KTDIF);
/* gauss-jordan solver external routine*/
AMREX_GPU_HOST_DEVICE void sgjsolve(double* A, double* x, double* b);
AMREX_GPU_HOST_DEVICE void sgjsolve_simplified(double* A, double* x, double* b);
}

#ifndef AMREX_USE_CUDA
namespace thermo
{

    extern double fwd_A[218], fwd_beta[218], fwd_Ea[218];
    extern double low_A[218], low_beta[218], low_Ea[218];
    extern double rev_A[218], rev_beta[218], rev_Ea[218];
    extern double troe_a[218],troe_Ts[218], troe_Tss[218], troe_Tsss[218];
    extern double sri_a[218], sri_b[218], sri_c[218], sri_d[218], sri_e[218];
    extern double activation_units[218], prefactor_units[218], phase_units[218];
    extern int is_PD[218], troe_len[218], sri_len[218], nTB[218], *TBid[218];
    extern double *TB[218];
    extern std::vector<std::vector<double>> kiv; 
    extern std::vector<std::vector<double>> nuv; 
    extern std::vector<std::vector<double>> kiv_qss; 
    extern std::vector<std::vector<double>> nuv_qss; 
}
#endif
