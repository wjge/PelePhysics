#ifndef _REACTOR_BASE_H_
#define _REACTOR_BASE_H_

#include <AMReX_Utility.H>

//using namespace amrex;

class ReactorBase
{
public:
    ReactorBase();
    virtual ~ReactorBase();

    virtual int reactor_init(int cvode_iE, int Ncells);

    virtual int react(amrex::Real *rY_in, amrex::Real *rY_src_in, 
       	      amrex::Real *rX_in, amrex::Real *rX_src_in, 
    	      amrex::Real &dt_react, amrex::Real &time);

    virtual int react(const amrex::Box& box,
                  amrex::Array4<amrex::Real> const& rY_in,
                  amrex::Array4<amrex::Real> const& rY_src_in, 
                  amrex::Array4<amrex::Real> const& T_in, 
                  amrex::Array4<amrex::Real> const& rEner_in,  
                  amrex::Array4<amrex::Real> const& rEner_src_in,
                  amrex::Array4<amrex::Real> const& FC_in,
                  amrex::Array4<int> const& mask, 
                  amrex::Real &dt_react,
                  amrex::Real &time);

    virtual void reactor_close();

    virtual int check_flag(void *flagvalue, const char *funcname, int opt);

    virtual void PrintFinalStats(void *cvodeMem, amrex::Real Temp);

    virtual void SetTypValsODE(const std::vector<double>& ExtTypVals);

    virtual void SetTolFactODE(double relative_tol,double absolute_tol);

};
#endif
