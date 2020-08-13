#ifndef _REACTOR_CVODECPU_H_
#define _REACTOR_CVODECPU_H_

#include <math.h>
#include <iostream>
#include <cstring>
#include <chrono>

#include <cvode/cvode.h>               /* prototypes for CVODE fcts., consts.  */
#include <nvector/nvector_serial.h>    /* access to serial N_Vector            */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix            */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver      */
#include <sunlinsol/sunlinsol_spgmr.h> /* access to SPGMR SUNLinearSolver     */
#include <cvode/cvode_direct.h>        /* access to CVDls interface            */
#include <cvode/cvode_spils.h>         /* access to CVSpils interface */
#include <sundials/sundials_types.h>   /* defs. of realtype, sunindextype      */
#include <sundials/sundials_math.h>
#include <sunmatrix/sunmatrix_sparse.h>

#ifdef USE_KLU_PP 
#include "klu.h"
#include <sunlinsol/sunlinsol_klu.h>
#endif

#include <AMReX_FArrayBox.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

#include <AMReX_Print.H>
#include <EOS.H>
#include <AMReX_ParmParse.H>
#include <chemistry_file.H>
#include "mechanism.h"

#include <ReactorBase.H>

class ReactorCVODE_CPU
    : public ReactorBase
{

    typedef struct {
          /* hacks */
          bool FirstTimePrecond;
          /* Checks */
          bool reactor_cvode_initialized;
          bool actual_ok_to_react;
          /* Base items */
          int ncells;
          int iverbose;
          int isolve_type;
          int ianalytical_jacobian;
          int ireactor_type;
          int boxcell;
          /* external forcing */
          amrex::Real *Yvect_full = NULL;
          amrex::Real *rhoX_init = NULL;
          amrex::Real *rhoXsrc_ext = NULL;
          amrex::Real *rYsrc = NULL;
          int *FCunt = NULL;
    
          /* Options */
          int NNZ; 
          /* Sparse Matrices for KLU-related solve */
          SUNMatrix *PS;
          /* SUNSparseMatrix_Data */
          realtype **Jdata  = NULL;
          /* SUNSparseMatrix_IndexValues */
          int **rowVals     = NULL;
          int **rowPtrs     = NULL;
          /* SUNSparseMatrix_IndexPointers */
          int **colPtrs     = NULL;
          int **colVals     = NULL;
          /* Holder for sparse matrix in Fuego fetches */
          int *indx = NULL;
          realtype **JSPSmat = NULL;
#ifdef USE_KLU_PP 
          /* KLU objects */
          klu_common *Common;
          klu_symbolic **Symbolic;
          klu_numeric **Numeric;
#else
          realtype **(**Jbd);
          realtype **(**P);
          sunindextype *(**pivot);
#endif
          /* Sparse custom */
          SUNMatrix PSc;
          int *colVals_c;
          int *rowPtrs_c;
    } *UserData;


/**********************************/
/* Functions Called by the Program */

    virtual int reactor_init(int cvode_iE, int Ncells) override;

    virtual int react(amrex::Real *rY_in, amrex::Real *rY_src_in, 
	          amrex::Real *rX_in, amrex::Real *rX_src_in, 
	          amrex::Real &dt_react, amrex::Real &time) override;

    virtual void reactor_close() override;

/**********************************/
/* Functions Called by the Solver */

    int cF_RHS(realtype t, N_Vector y_in, N_Vector ydot, void *user_data);

/**********************************/
/* Helper functions */
    virtual int check_flag(void *flagvalue, const char *funcname, int opt) override;

    virtual void SetTypValsODE(const std::vector<double>& ExtTypVals) override;

    UserData AllocUserData(int iE, int num_cells);

}
#endif
