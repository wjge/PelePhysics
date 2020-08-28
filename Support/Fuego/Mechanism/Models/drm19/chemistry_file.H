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
AMREX_GPU_HOST_DEVICE void gibbs(double *  species, double *  tc);
AMREX_GPU_HOST_DEVICE void helmholtz(double *  species, double *  tc);
AMREX_GPU_HOST_DEVICE void speciesInternalEnergy(double *  species, double *  tc);
AMREX_GPU_HOST_DEVICE void speciesEnthalpy(double *  species, double *  tc);
AMREX_GPU_HOST_DEVICE void speciesEntropy(double *  species, double *  tc);
AMREX_GPU_HOST_DEVICE void cp_R(double *  species, double *  tc);
AMREX_GPU_HOST_DEVICE void cv_R(double *  species, double *  tc);
AMREX_GPU_HOST_DEVICE void productionRate(double *  wdot, double *  sc, double T);
AMREX_GPU_HOST_DEVICE void comp_qfqr(double *  q_f, double *  q_r, double *  sc, double *  tc, double invT);
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
/*gauss-jordan solver external routine */
AMREX_GPU_HOST_DEVICE void sgjsolve(double* A, double* x, double* b);
AMREX_GPU_HOST_DEVICE void sgjsolve_simplified(double* A, double* x, double* b);
}

namespace thermo
{

    extern AMREX_GPU_DEVICE_MANAGED double fwd_A[84], fwd_beta[84], fwd_Ea[84];
    extern AMREX_GPU_DEVICE_MANAGED double low_A[84], low_beta[84], low_Ea[84];
    extern AMREX_GPU_DEVICE_MANAGED double rev_A[84], rev_beta[84], rev_Ea[84];
    extern AMREX_GPU_DEVICE_MANAGED double troe_a[84],troe_Ts[84], troe_Tss[84], troe_Tsss[84];
    extern AMREX_GPU_DEVICE_MANAGED double sri_a[84], sri_b[84], sri_c[84], sri_d[84], sri_e[84];
    extern AMREX_GPU_DEVICE_MANAGED double activation_units[84], prefactor_units[84], phase_units[84];
    extern AMREX_GPU_DEVICE_MANAGED int is_PD[84], troe_len[84], sri_len[84], nTB[84], *TBid[84];
    extern AMREX_GPU_DEVICE_MANAGED double *TB[84];

    extern std::vector<std::vector<int>> kiv; 
    extern std::vector<std::vector<int>> nuv; 

    extern AMREX_GPU_DEVICE_MANAGED int ki2D[420], nu2D[420];
}
