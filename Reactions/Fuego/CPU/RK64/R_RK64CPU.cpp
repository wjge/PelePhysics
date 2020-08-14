#include <R_RK64CPU.H>

using namespace amrex;

int ReactorRK64_CPU::eint_rho;
int ReactorRK64_CPU::enth_rho;

/**********************************/
/* Class functions */
ReactorRK64_CPU::ReactorRK64_CPU() 
{ 
    data       = NULL;
    // Scaling
    typVals = {-1};
    relTol  = 1.0e-10;
    absTol  = 1.0e-10;
    /* OPTIONS -- should be static */
    ReactorRK64_CPU::eint_rho = 1; // in/out = rhoE/rhoY
    ReactorRK64_CPU::enth_rho = 2; // in/out = rhoH/rhoY 
}

ReactorRK64_CPU::~ReactorRK64_CPU() { ; }

/**********************************/
/* Functions Called by the Program */

/* Initialization routine, called once at the begining of the problem */
int ReactorRK64_CPU::reactor_init(int reactor_type, int ode_ncells,
                                  double rk64_errtol, int rk64_nsubsteps_guess,
                                  int rk64_nsubsteps_min,int rk64_nsubsteps_max) {

    int omp_thread = 0;
#ifdef _OPENMP
    omp_thread = omp_get_thread_num(); 
#endif
    data = AllocUserData(reactor_type, ode_ncells, 
                         rk64_errtol, rk64_nsubsteps_guess,
                         rk64_nsubsteps_min, rk64_nsubsteps_max);

    /* Number of species and cells in mechanism */
    if ((data->iverbose > 0) && (omp_thread == 0)) {
        Print() << "Number of species in mech is " << NUM_SPECIES << "\n";
        Print() << "Number of cells in one solve is " << data->ncells << "\n";
    }

    return(0);
}

/* Initialization routine, called once at the begining of the problem */
int ReactorRK64_CPU::reactor_init(int reactor_type, int ode_ncells) {

    double rk64_errtol       = 1e-16;
    int rk64_nsubsteps_guess = 10; 
    int rk64_nsubsteps_min   = 5;
    int rk64_nsubsteps_max   = 500;

    int omp_thread = 0;
#ifdef _OPENMP
    /* omp thread if applicable */
    omp_thread = omp_get_thread_num(); 
#endif
    data = AllocUserData(reactor_type, ode_ncells,
                         rk64_errtol, rk64_nsubsteps_guess,
                         rk64_nsubsteps_min, rk64_nsubsteps_max);

    /* Number of species and cells in mechanism */
    if ((data->iverbose > 0) && (omp_thread == 0)) {
        Print() << "Number of species in mech is " << NUM_SPECIES << "\n";
        Print() << "Number of cells in one solve is " << data->ncells << "\n";
    }

    return(0);
}

/* Main call routine */
int ReactorRK64_CPU::react(double *rY_in, double *rY_src_in, 
                           double *rX_in, double *rX_src_in,
                           double &dt_react, double &time)
{

    double time_init = time;
    double time_out  = time + dt_react;

    double current_time = time_init;

    double *soln_reg,*carryover_reg,*error_reg,*zero_reg,*rhs;
    double dt_rk,dt_rk_min,dt_rk_max,change_factor;

    const double exp1=0.25;
    const double exp2=0.2;
    const double beta=1.0;
    const double tinyval=1e-50;

    int neq_tot        = (NUM_SPECIES + 1) * (data->ncells);

    const int nstages_rk64=6;
    const amrex::Real alpha_rk64[6] = {
        0.218150805229859,  //            3296351145737.0/15110423921029.0,
        0.256702469801519,  //            1879360555526.0/ 7321162733569.0,
        0.527402592007520,  //            10797097731880.0/20472212111779.0,
        0.0484864267224467, //            754636544611.0/15563872110659.0,
        1.24517071533530,   //            3260218886217.0/ 2618290685819.0,
        0.412366034843237,  //            5069185909380.0/12292927838509.0
    };

    const amrex::Real beta_rk64[6] = {
        -0.113554138044166,  //-1204558336989.0/10607789004752.0,
        -0.215118587818400,  //-3028468927040.0/14078136890693.0,
        -0.0510152146250577, //-455570672869.0/ 8930094212428.0,
        -1.07992686223881,   //-17275898420483.0/15997285579755.0,
        -0.248664241213447,  //-2453906524165.0/ 9868353053862.0,
        0.0};

    const amrex::Real err_rk64[6] = {
        -0.0554699315064507, //-530312978447.0/ 9560368366154.0,
        0.158481845574980,   // 473021958881.0/ 2984707536468.0,
        -0.0905918835751907, //-947229622805.0/10456009803779.0,
        -0.219084567203338,  //-2921473878215.0/13334914072261.0,
        0.164022338959433,   // 1519535112975.0/ 9264196100452.0,
        0.0426421977505659   // 167623581683.0/ 3930932046784.0
    };

    int omp_thread = 0;
#ifdef _OPENMP
    /* omp thread if applicable */
    omp_thread = omp_get_thread_num(); 
#endif

    if ((data->iverbose > 1) && (omp_thread == 0)) 
    {
	    amrex::Print() <<"\n -------------------------------------\n";
    }

    if ((data->iverbose > 3) && (omp_thread == 0)) 
    {
	amrex::Print() <<"BEG : time curr is "<< time_init << 
            " and dt_react is " << dt_react << 
            " and final time should be " << time_out << "\n";
    }

    soln_reg      = (double *) calloc(neq_tot,sizeof(double));
    carryover_reg = (double *) calloc(neq_tot,sizeof(double));
    error_reg     = (double *) calloc(neq_tot,sizeof(double));
    zero_reg      = (double *) calloc(neq_tot,sizeof(double));
    rhs           = (double *) calloc(neq_tot,sizeof(double));


    /* Get Device MemCpy of in arrays */
    /* Get Device pointer of solution vector */
    /* rhoY,T */
    std::memcpy(soln_reg,      rY_in, sizeof(double) * neq_tot);
    std::memcpy(carryover_reg, rY_in, sizeof(double) * neq_tot);
    /* rhoY_src_ext */
    std::memcpy(data->rYsrc, rY_src_in, (NUM_SPECIES * data->ncells)*sizeof(double));
    /* rhoE/rhoH */
    std::memcpy(data->rhoX_init, rX_in, sizeof(double) * data->ncells);
    std::memcpy(data->rhoXsrc_ext, rX_src_in, sizeof(double) * data->ncells);

    dt_rk     = dt_react/double(data->nsubsteps_guess);
    dt_rk_min = dt_react/double(data->nsubsteps_max);
    dt_rk_max = dt_react/double(data->nsubsteps_min);

    int nsteps=0;

    while(current_time < time_out)
    {
        current_time += dt_rk;
        nsteps++;
        std::memcpy(carryover_reg, soln_reg, sizeof(double) * neq_tot);
        std::memcpy(error_reg,     zero_reg, sizeof(double) * neq_tot);

        for(int stage=0;stage<nstages_rk64;stage++)
        {
            std::memcpy(rhs,zero_reg, sizeof(double) * neq_tot);
            cF_RHS(current_time, soln_reg, rhs, data);

            for(int i=0;i<neq_tot;i++)
            {
                error_reg[i]    += err_rk64[stage]*dt_rk*rhs[i];
                soln_reg[i]      = carryover_reg[i] + alpha_rk64[stage]*dt_rk*rhs[i];
                carryover_reg[i] = soln_reg[i]      + beta_rk64[stage] *dt_rk*rhs[i];
            }
        } 

        //adapt time-step
        double max_err=tinyval;
        for(int i=0;i<neq_tot;i++)
        {
            if(fabs(error_reg[i]) > max_err)
            {
                max_err=fabs(error_reg[i]);
            }
        }

        //chance to increase time step
        if(max_err < data->errtol)
        {
            change_factor = beta*pow((data->errtol/max_err),exp1);
            dt_rk = std::min(dt_rk_max,dt_rk*change_factor);
        }
        //reduce time step (error is high!)
        else
        {
            change_factor=beta*pow((data->errtol/max_err),exp2);
            dt_rk = std::max(dt_rk_min,dt_rk*change_factor);
        }
    }

    //update guess from the current update
    (data->nsubsteps_guess) = nsteps;

#ifdef MOD_REACTOR
    /* If reactor mode is activated, update time */
    time  = time_init + dt_react;
#endif

    if ((data->iverbose > 3) && (omp_thread == 0)) 
    {
          amrex::Print() <<"END : time curr is "<< current_time << 
              " and actual dt_react is " << dt_react << "\n";
    }

    /* Pack data to return in main routine external */
    std::memcpy(rY_in, soln_reg, neq_tot * sizeof(double));
    for  (int i = 0; i < data->ncells; i++) 
    {
        rX_in[i] = rX_in[i] + (dt_react) * rX_src_in[i];
    }

    return nsteps;
}

/* Free memory */
void ReactorRK64_CPU::reactor_close()
{
    FreeUserData(data);
}
/**********************************/

/**********************************/
/* Functions Called by the Solver */

/**********************************/

/**********************************/
/* Helper functions */

/* Set or update typVals */
void ReactorRK64_CPU::SetTypValsODE(const std::vector<double>& ExtTypVals) {
    int size_ETV = (NUM_SPECIES + 1);
    Vector<std::string> kname;
    EOS::speciesNames(kname);
    int omp_thread = 0;
#ifdef _OPENMP
    omp_thread = omp_get_thread_num();
#endif

    for (int i=0; i<size_ETV-1; i++) {
      typVals[i] = ExtTypVals[i];
    }
    typVals[size_ETV-1] = ExtTypVals[size_ETV-1];
    if (omp_thread == 0){
        Print() << "Set the typVals in PelePhysics: \n  ";
        for (int i=0; i<size_ETV-1; i++) {
            Print() << kname[i] << ":" << typVals[i] << "  ";
        }
        Print() << "Temp:"<< typVals[size_ETV-1] <<  " \n";
    }
}

/* Set or update the rel/abs tolerances  */
void ReactorRK64_CPU::SetTolFactODE(double relative_tol,double absolute_tol) {
    relTol = relative_tol;
    absTol = absolute_tol;
    int omp_thread = 0;
#ifdef _OPENMP
    omp_thread = omp_get_thread_num();
#endif

    if (omp_thread == 0){
        Print() << "Set RTOL, ATOL = "<<relTol<< " "<<absTol<<  " in PelePhysics\n";
    }
}

ReactorRK64_CPU::UserData ReactorRK64_CPU::AllocUserData(int reactor_type, int num_cells,
                                                   double rk64_errtol, int rk64_nsubsteps_guess,
                                                   int rk64_nsubsteps_min, int rk64_nsubsteps_max)
{
    /* Make local copies of pointers in user_data */
    UserData data_wk = (UserData) malloc(sizeof *data_wk);
    int omp_thread = 0;
#ifdef _OPENMP
    omp_thread = omp_get_thread_num(); 
#endif

    (data_wk->ncells)        = num_cells;
    (data_wk->iverbose)      = 1;
    (data_wk->ireactor_type) = reactor_type;

    (data_wk->errtol) = rk64_errtol;
    (data_wk->nsubsteps_guess) = rk64_nsubsteps_guess;
    (data_wk->nsubsteps_min)   = rk64_nsubsteps_min;
    (data_wk->nsubsteps_max)   = rk64_nsubsteps_max;

    (data_wk->rYsrc)       = new  amrex::Real[data_wk->ncells*(NUM_SPECIES)];
    (data_wk->rhoX_init)   = new  amrex::Real[data_wk->ncells];
    (data_wk->rhoXsrc_ext) = new  amrex::Real[data_wk->ncells];

    return(data_wk);
}

/* Free data memory */
void ReactorRK64_CPU::FreeUserData(UserData data_wk)
{
    delete[] (data_wk->rYsrc);
    delete[] (data_wk->rhoX_init);
    delete[] (data_wk->rhoXsrc_ext);

    free(data_wk);
} 
/******************************************************************************************/
/* RHS source terms evaluation */
void ReactorRK64_CPU::cF_RHS(double &t, double *yvec_d, double *ydot_d,  
        void *user_data)
{
    /* Make local copies of pointers in user_data (cell M)*/
    UserData data_wk;
    data_wk = (UserData) user_data;   

    /* Tmp vars */
    int tid;

    /* Loop on packed cells */
    for (tid = 0; tid < data_wk->ncells; tid ++) 
    {
        /* Tmp vars */
        double massfrac[NUM_SPECIES];
        double Xi[NUM_SPECIES];
        double cdot[NUM_SPECIES], molecular_weight[NUM_SPECIES];
        double cX;
        double temp, energy;

        /* Offset in case several cells */
        int offset = tid * (NUM_SPECIES + 1); 

        /* MW CGS */
        CKWT(molecular_weight);

        /* rho MKS */ 
        double rho = 0.0;
        for (int i = 0; i < NUM_SPECIES; i++)
        {
            rho = rho + yvec_d[offset + i];
        }

        /* temp */
        temp = yvec_d[offset + NUM_SPECIES];

        /* Yks */
        for (int i = 0; i < NUM_SPECIES; i++)
        {
            massfrac[i] = yvec_d[offset + i] / rho;
        }
        

        /* NRG CGS */
        energy = (data_wk->rhoX_init[tid] + data_wk->rhoXsrc_ext[tid] * t) /rho;

        if (data_wk->ireactor_type == eint_rho)
        {
            /* UV REACTOR */
            EOS::EY2T(energy, massfrac, temp);
            EOS::TY2Cv(temp, massfrac, cX);
            EOS::T2Ei(temp, Xi);
        } 
        else 
        {
            /* HP REACTOR */
            EOS::HY2T(energy, massfrac, temp);
            EOS::TY2Cp(temp, massfrac, cX);
            EOS::T2Hi(temp, Xi);
        }
        EOS::RTY2WDOT(rho, temp, massfrac, cdot);

        /* Fill ydot vect */
        ydot_d[offset + NUM_SPECIES] = data_wk->rhoXsrc_ext[tid];
        for (int i = 0; i < NUM_SPECIES; i++)
        {
            ydot_d[offset + i] = cdot[i] + data_wk->rYsrc[tid * (NUM_SPECIES) + i];
            ydot_d[offset + NUM_SPECIES] = ydot_d[offset + NUM_SPECIES]  - ydot_d[offset + i] * Xi[i];
        }
        ydot_d[offset + NUM_SPECIES] = ydot_d[offset + NUM_SPECIES] /(rho * cX);
    }
}

/* End of file  */
