#ifndef _REACTOR_ARKODECPU_H_
#define _REACTOR_ARKODECPU_H_

#include <math.h>
#include <iostream>
#include <cstring>
#include <chrono>

#include <arkode/arkode_arkstep.h>
#include <arkode/arkode_erkstep.h>

#include <nvector/nvector_serial.h>    /* access to serial N_Vector            */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix            */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver      */
#include <sunnonlinsol/sunnonlinsol_newton.h>
#include <sundials/sundials_types.h>   /* defs. of realtype, sunindextype      */
#include <sundials/sundials_math.h>

#include <AMReX_Print.H>
#include <EOS.H>

#include <AMReX_ParmParse.H>
#include <chemistry_file.H>
#include "mechanism.h"
#include <AMREX_misc.H>

#include <ReactorBase.H>

using namespace amrex;

class ReactorARKODE_CPU
    : public ReactorBase
{

public:

/**********************************/
/* main CVODE helper */
    typedef struct {
      /* Checks */
      bool reactor_arkode_initialized;
      /* Base items */
      int ncells;
      int iverbose;
      int ianalytical_jacobian;
      int ireactor_type;
      int iimplicit_solve;
      int iuse_erkode;
      /* external forcing */
      //amrex::Real *Yvect_full = NULL;
      //amrex::Real *rhoX_init = NULL;
      //amrex::Real *rhoXsrc_ext = NULL;
      //amrex::Real *rYsrc = NULL;
      //int *FCunt = NULL;
      //int *mask = NULL;
    } *UserData;

/**********************************/
/* Global Variables */
    // ARKODE
    N_Vector y;
    SUNLinearSolver LS;
    SUNNonlinearSolver NLS;
    SUNMatrix A;
    void *arkode_mem;
    UserData data;
    // Scaling
    Array<double,NUM_SPECIES+1> typVals;
    double relTol;
    double absTol;
    /* energy -- Should be moved in struct */
    double *rhoX_init;
    double *rhoXsrc_ext;
    double *rYsrc;
    /* OPTIONS -- should be static */
    int dense_solve;
    static int eint_rho; // in/out = rhoE/rhoY
    static int enth_rho; // in/out = rhoH/rhoY 


/**********************************/
/* Class functions */
    ReactorARKODE_CPU();
    virtual ~ReactorARKODE_CPU();

/**********************************/
/* Functions Called by the Solver */
    static int cF_RHS(realtype t, N_Vector y_in, N_Vector ydot, void *user_data);

    static int cJac(realtype tn, N_Vector y, N_Vector fy, SUNMatrix J,
		void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

/**********************************/
/* Functions Called by the Program */

    virtual int reactor_init(int cvode_iE, int Ncells) override;

    virtual int react(realtype *rY_in, realtype *rY_src_in,
              realtype *rX_in, realtype *rX_src_in,
              realtype &dt_react, realtype &time) override;

    virtual void reactor_close() override;

/**********************************/
/* Helper functions */

    virtual void PrintFinalStats(void *arkodeMem, realtype Temp) override;

    virtual int check_flag(void *flagvalue, const char *funcname, int opt) override;

    virtual void SetTypValsODE(const std::vector<double>& ExtTypVals) override;

    virtual void SetTolFactODE(double relative_tol,double absolute_tol) override;

    void ReSetTolODE();

    UserData AllocUserData(int iE, int num_cells);

    void FreeUserData(UserData data);

///**********************************/
///* Main Kernel fct called in solver RHS */
//void fKernelSpec(realtype *dt, realtype *yvec_d, realtype *ydot_d,
//		void *user_data);

};

#endif
