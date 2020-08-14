#ifndef _REACTOR_RK64CPU_H
#define _REACTOR_RK64CPU_H

#include <math.h>
#include <iostream>
#include <cstring>
#include <chrono>

#include <AMReX_Print.H>
#include <EOS.H>

#include <AMReX_ParmParse.H>
#include <chemistry_file.H>
#include "mechanism.h"

#include <ReactorBase.H>

using namespace amrex;

class ReactorRK64_CPU
    : public ReactorBase
{

public:

/**********************************/
/* main RK64 helper */
    typedef struct {
        /* Checks */
        /* Base items */
        int ncells;
        int iverbose;
        int ireactor_type;
        double errtol;
        int nsubsteps_guess;
        int nsubsteps_min;
        int nsubsteps_max;
        /* external forcing */
        amrex::Real *rhoX_init = NULL;
        amrex::Real *rhoXsrc_ext = NULL;
        amrex::Real *rYsrc = NULL;
    } *UserData;

/**********************************/
/* Global Variables */
    UserData data;
    // Scaling
    Array<double,NUM_SPECIES+1> typVals;
    double relTol;
    double absTol;
    /* energy */
    static int eint_rho; // in/out = rhoE/rhoY
    static int enth_rho; // in/out = rhoH/rhoY 


/**********************************/
/* Class functions */
    ReactorRK64_CPU();
    virtual ~ReactorRK64_CPU();

/**********************************/
/* Functions Called by the Solver */

/**********************************/
/* Functions Called by the Program */

    virtual int reactor_init(int reactor_type, int Ncells) override;

    virtual int react(amrex::Real *rY_in, amrex::Real *rY_src_in, 
              amrex::Real *rX_in, amrex::Real *rX_src_in, 
              amrex::Real &dt_react, amrex::Real &time) override;

    virtual void reactor_close() override;

/**********************************/
/* Helper functions */

    void cF_RHS(double &dt, double *yvec_d, double *ydot_d,
            void *user_data);

    virtual void SetTypValsODE(const std::vector<double>& ExtTypVals) override;

    virtual void SetTolFactODE(double relative_tol,double absolute_tol) override;

    UserData AllocUserData(int reactor_type, int num_cells);

    void FreeUserData(UserData data);
};

#endif
