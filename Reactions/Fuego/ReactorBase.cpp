#include <ReactorBase.H>

ReactorBase::ReactorBase() { ; }
ReactorBase::~ReactorBase(){ ; }  

int ReactorBase::reactor_init(int cvode_iE, int Ncells)
{
    amrex::Error("ReactorBase::reactor_init not defined");
    return(0);
}

int ReactorBase::react(amrex::Real *rY_in, amrex::Real *rY_src_in,
                       amrex::Real *rX_in, amrex::Real *rX_src_in,
                       amrex::Real &dt_react, amrex::Real &time) 
{
    amrex::Error("ReactorBase::react not defined");
    return(0);
}

int ReactorBase::react(const amrex::Box& box,
                  amrex::Array4<amrex::Real> const& rY_in,
                  amrex::Array4<amrex::Real> const& rY_src_in, 
                  amrex::Array4<amrex::Real> const& T_in, 
                  amrex::Array4<amrex::Real> const& rEner_in,  
                  amrex::Array4<amrex::Real> const& rEner_src_in,
                  amrex::Array4<amrex::Real> const& FC_in,
                  amrex::Array4<int> const& mask, 
                  amrex::Real &dt_react,
                  amrex::Real &time)
{
    amrex::Error("ReactorBase::react not defined");
    return(0);
}

void ReactorBase::reactor_close()
{
    amrex::Error("ReactorBase::reactor_close not defined");
}

int ReactorBase::check_flag(void *flagvalue, const char *funcname, int opt)
{
    amrex::Error("ReactorBase::check_flag not defined");
    return(0);
}

void ReactorBase::PrintFinalStats(void *cvodeMem, amrex::Real Temp)
{
    amrex::Error("ReactorBase::PrintFinalStats not defined");
}

void ReactorBase::SetTypValsODE(const std::vector<double>& ExtTypVals)
{
    amrex::Error("ReactorBase::SetTypValsODE not defined");
}

void ReactorBase::SetTolFactODE(double relative_tol,double absolute_tol)
{
    amrex::Error("ReactorBase::SetTolFactODE not defined");
}

