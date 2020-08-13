#ifndef _REACTOR_BASE_H_
#define _REACTOR_BASE_H_

class ReactorBase
{

    virtual int reactor_init(int cvode_iE, int Ncells);

    virtual int react(amrex::Real *rY_in, amrex::Real *rY_src_in, 
       	      amrex::Real *rX_in, amrex::Real *rX_src_in, 
    	      amrex::Real &dt_react, amrex::Real &time);

    virtual void reactor_close();

    virtual int check_flag(void *flagvalue, const char *funcname, int opt);

    virtual void SetTypValsODE(const std::vector<double>& ExtTypVals);

}
#endif
