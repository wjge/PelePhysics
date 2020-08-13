#include <R_CvodeCPU.H>

using namespace amrex;

/**********************************/
/* Functions Called by the Program */

/* Initialization routine, called once at the begining of the problem */
void ReactorCVODE_CPU::reactor_init(int reactor_type, int ode_ncells)
{

    BL_PROFILE_VAR("reactInit", reactInit);

    int omp_thread = 0;
#ifdef _OPENMP
    omp_thread = omp_get_thread_num();
#endif
    /* Total number of eq to integrate */
    int neq_tot = (NUM_SPECIES + 1) * ode_ncells;

    /* Definition of main vector */
    y = N_VNew_Serial(neq_tot);
    if (check_flag((void *)y, "N_VNew_Serial", 0)) return(1);

    /* Call CVodeCreate to create the solver memory and specify the
     * Backward Differentiation Formula and the use of a Newton iteration */
    cvode_mem = CVodeCreate(CV_BDF);
    if (check_flag((void *)cvode_mem, "CVodeCreate", 0)) return(1);

    /* Does not work for more than 1 cell right now */
    data = AllocUserData(reactor_type, ode_ncells);
    if(check_flag((void *)data, "AllocUserData", 2)) return(1);

    /* Number of species and cells in mechanism */
    if ((data->iverbose > 0) && (omp_thread == 0)) {
        Print() << "Number of species in mech is " << NUM_SPECIES << "\n";
        Print() << "Number of cells in one solve is " << data->ncells << "\n";
    }

    /* Set the pointer to user-defined data */
    int flag = CVodeSetUserData(cvode_mem, data);
    if(check_flag(&flag, "CVodeSetUserData", 1)) return(1);   

    realtype time = 0.0e+0;
    /* Call CVodeInit to initialize the integrator memory and specify the
     * user's right hand side function, the inital time, and 
     * initial dependent variable vector y. */
    flag = CVodeInit(cvode_mem, cF_RHS, time, y);
    if (check_flag(&flag, "CVodeInit", 1)) return(1);
    
    /* Definition of tolerances: one for each species */
    N_Vector atol = N_VNew_Serial(neq_tot);
    realtype *ratol = N_VGetArrayPointer(atol);
    if (typVals[0] > 0) {
        if ((data->iverbose > 0) && (omp_thread == 0)) {
            Print() << "Setting CVODE tolerances rtol = " << relTol << " atolfact = " << absTol << " in PelePhysics \n";
        }
        for  (int i = 0; i < data->ncells; i++) {
            int offset = i * (NUM_SPECIES + 1);
            for  (int k = 0; k < NUM_SPECIES + 1; k++) {
                //ratol[offset + k] = std::max(typVals[k]*absTol,relTol);
                ratol[offset + k] = typVals[k]*absTol;
            }
        }
    } else {
        if ((data->iverbose > 0) && (omp_thread == 0)) {
            Print() << "Setting CVODE tolerances rtol = " << relTol << " atol = " << absTol << " in PelePhysics \n";
        }
        for (int i=0; i<neq_tot; i++) {
            ratol[i] = absTol;
        }
    }
    /* Call CVodeSVtolerances to specify the scalar relative tolerance
     * and vector absolute tolerances */
    flag = CVodeSVtolerances(cvode_mem, relTol, atol);
    if (check_flag(&flag, "CVodeSVtolerances", 1)) return(1);

    flag = CVodeSetMaxNonlinIters(cvode_mem, 50);
    if (check_flag(&flag, "CVodeSetMaxNonlinIters", 1)) return(1);

    flag = CVodeSetMaxErrTestFails(cvode_mem, 100);
    if (check_flag(&flag, "CVodeSetMaxErrTestFails", 1)) return(1);

    if (data->isolve_type == dense_solve) {
        if ((data->iverbose > 0) && (omp_thread == 0)) {
            Print() << "\n--> Using a Direct Dense Solver\n";
        }
        /* Create dense SUNMatrix for use in linear solves */
        A = SUNDenseMatrix(neq_tot, neq_tot);
        if(check_flag((void *)A, "SUNDenseMatrix", 0)) return(1);

        /* Create dense SUNLinearSolver object for use by CVode */
        LS = SUNDenseLinearSolver(y, A);
        if(check_flag((void *)LS, "SUNDenseLinearSolver", 0)) return(1);

        /* Call CVDlsSetLinearSolver to attach the matrix and linear solver to CVode */
        flag = CVDlsSetLinearSolver(cvode_mem, LS, A);
        if(check_flag(&flag, "CVDlsSetLinearSolver", 1)) return(1);

    } else if (data->isolve_type == sparse_solve_custom) {
        if ((data->iverbose > 0) && (omp_thread == 0)) {
            Print() << "\n--> Using a custom Direct Sparse Solver\n";
        }
        /* Create dense SUNMatrix for use in linear solves */
        A = SUNSparseMatrix(neq_tot, neq_tot, (data->NNZ)*data->ncells, CSR_MAT);
        if(check_flag((void *)A, "SUNDenseMatrix", 0)) return(1);

        /* Create dense SUNLinearSolver object for use by CVode */
        LS = SUNLinSol_sparse_custom(y, A, reactor_type, data->ncells, (NUM_SPECIES+1), data->NNZ);
        if(check_flag((void *)LS, "SUNDenseLinearSolver", 0)) return(1);

        /* Call CVDlsSetLinearSolver to attach the matrix and linear solver to CVode */
        flag = CVDlsSetLinearSolver(cvode_mem, LS, A);
        if(check_flag(&flag, "CVDlsSetLinearSolver", 1)) return(1);

    } else if (data->isolve_type == sparse_solve) {
#ifdef USE_KLU_PP 
        if ((data->iverbose > 0) && (omp_thread == 0)) {
            Print() << "\n--> Using a Direct Sparse Solver\n";
        }
        /* Create sparse SUNMatrix for use in linear solves */
        A = SUNSparseMatrix(neq_tot, neq_tot, (data->NNZ)*data->ncells, CSC_MAT);
        if(check_flag((void *)A, "SUNSparseMatrix", 0)) return(1);

        /* Create KLU solver object for use by CVode */
        LS = SUNLinSol_KLU(y, A);
        if(check_flag((void *)LS, "SUNLinSol_KLU", 0)) return(1);

        /* Call CVodeSetLinearSolver to attach the matrix and linear solver to CVode */
        flag = CVodeSetLinearSolver(cvode_mem, LS, A);
        if(check_flag(&flag, "CVodeSetLinearSolver", 1)) return(1);
#else        
        Abort("Sparse solver not valid without KLU solver.");
#endif

    } else if ((data->isolve_type == iterative_gmres_solve) 
            || (data->isolve_type == iterative_gmres_solve_custom)) {
        if ((data->iverbose > 0) && (omp_thread == 0)) {
            Print() << "\n--> Using an Iterative Solver ("<<data->isolve_type<<")\n";
        }

        /* Create the linear solver object */
        if (data->ianalytical_jacobian == 0) {
            LS = SUNSPGMR(y, PREC_NONE, 0);
            if(check_flag((void *)LS, "SUNDenseLinearSolver", 0)) return(1);
        } else {
            LS = SUNSPGMR(y, PREC_LEFT, 0);
            if(check_flag((void *)LS, "SUNDenseLinearSolver", 0)) return(1);
        }

        /* Set CVSpils linear solver to LS */
        flag = CVSpilsSetLinearSolver(cvode_mem, LS);
        if(check_flag(&flag, "CVSpilsSetLinearSolver", 1)) return(1);
    } else {
        Abort("Wrong choice of linear solver...");
    }

    if (data->ianalytical_jacobian == 0) {
        if ((data->iverbose > 0) && (omp_thread == 0)) {
            Print() << "    Without Analytical J/Preconditioner\n";
        }
#ifdef USE_KLU_PP 
        if (data->isolve_type == sparse_solve) {
            Abort("Sparse Solver requires an Analytical J");
        }
#endif
        if (data->isolve_type == sparse_solve_custom) {
            Abort("Custom sparse solver requires an Analytical J");
        }
    } else {
        if (data->isolve_type == iterative_gmres_solve_custom) {
            /* Set the JAcobian-times-vector function */
            flag = CVSpilsSetJacTimes(cvode_mem, NULL, NULL);
            if(check_flag(&flag, "CVSpilsSetJacTimes", 1)) return(1);

            if ((data->iverbose > 0) && (omp_thread == 0)) {
                Print() << "    With a custom Sparse Preconditioner\n";
            }
            /* Set the preconditioner solve and setup functions */
            flag = CVSpilsSetPreconditioner(cvode_mem, Precond_custom, PSolve_custom);
            if(check_flag(&flag, "CVSpilsSetPreconditioner", 1)) return(1);

        } else if (data->isolve_type == iterative_gmres_solve) {
            /* Set the JAcobian-times-vector function */
            flag = CVSpilsSetJacTimes(cvode_mem, NULL, NULL);
            if(check_flag(&flag, "CVSpilsSetJacTimes", 1)) return(1);
#ifdef USE_KLU_PP 
            if ((data->iverbose > 0) && (omp_thread == 0)) {
                Print() << "    With a Sparse Preconditioner\n";
            }
            /* Set the preconditioner solve and setup functions */
            flag = CVSpilsSetPreconditioner(cvode_mem, Precond_sparse, PSolve_sparse);
            if(check_flag(&flag, "CVSpilsSetPreconditioner", 1)) return(1);
#else
            if ((data->iverbose > 0) && (omp_thread == 0)) {
                Print() << "    With a Preconditioner\n";
            }
            /* Set the preconditioner solve and setup functions */
            flag = CVSpilsSetPreconditioner(cvode_mem, Precond, PSolve);
            if(check_flag(&flag, "CVSpilsSetPreconditioner", 1)) return(1);
#endif
#ifdef USE_KLU_PP 
        } else if (data->isolve_type == sparse_solve){
            if ((data->iverbose > 0) && (omp_thread == 0)) {
                Print() << "    With a Sparse Analytical J\n";
            }
            /* Set the user-supplied Jacobian routine Jac */
            flag = CVodeSetJacFn(cvode_mem, cJac_KLU);
            if(check_flag(&flag, "CVodeSetJacFn", 1)) return(1); 
#endif
        } else if (data->isolve_type == dense_solve){
            if ((data->iverbose > 0) && (omp_thread == 0)) {
                Print() << "    With Analytical J\n";
            }
            /* Set the user-supplied Jacobian routine Jac */
            flag = CVodeSetJacFn(cvode_mem, cJac);
            if(check_flag(&flag, "CVodeSetJacFn", 1)) return(1);

        } else if (data->isolve_type == sparse_solve_custom) {
            if ((data->iverbose > 0) && (omp_thread == 0)) {
                Print() << "    With a Sparse Analytical J\n";
            }
            /* Set the user-supplied Jacobian routine Jac */
            flag = CVodeSetJacFn(cvode_mem, cJac_sps);
            if(check_flag(&flag, "CVodeSetJacFn", 1)) return(1);
        }
    }

    /* Set the max number of time steps */ 
    flag = CVodeSetMaxNumSteps(cvode_mem, 10000);
    if(check_flag(&flag, "CVodeSetMaxNumSteps", 1)) return(1);

    /* Set the max order */ 
    flag = CVodeSetMaxOrd(cvode_mem, 2);
    if(check_flag(&flag, "CVodeSetMaxOrd", 1)) return(1);

    /* Set the num of steps to wait inbetween Jac evals */ 
    flag = CVodeSetMaxStepsBetweenJac(cvode_mem, 100);
    if(check_flag(&flag, "CVodeSetMaxStepsBetweenJac", 1)) return(1);

    /* Free the atol vector */
    N_VDestroy(atol);

    /* Ok we're done ...*/
    if ((data->iverbose > 1) && (omp_thread == 0)) {
        Print() << "\n--> DONE WITH INITIALIZATION (CPU)" << data->ireactor_type << "\n";
    }

    /* Reactor is now initialized */
    data->reactor_cvode_initialized = true;

    BL_PROFILE_VAR_STOP(reactInit);

    return(0);

}

/* Main routine for CVode integration: classic version */
int ReactorCVODE_CPU::react(realtype *rY_in, realtype *rY_src_in, 
          realtype *rX_in, realtype *rX_src_in,
          realtype &dt_react, realtype &time){

    realtype dummy_time;
    int flag;
    int omp_thread = 0;
#ifdef _OPENMP
    omp_thread = omp_get_thread_num();
#endif

    if ((data->iverbose > 1) && (omp_thread == 0)) {
        Print() <<"\n -------------------------------------\n";
    }

    /* Initial time and time to reach after integration */
    time_init = time;
    realtype time_out  = time + dt_react;

    if ((data->iverbose > 3) && (omp_thread == 0)) {
        Print() <<"BEG : time curr is "<< time_init << " and dt_react is " << dt_react << " and final time should be " << time_out << "\n";
    }

    /* Define full box_ncells length vectors to be integrated piece by piece
       by CVode */
    if ((data->iverbose > 2) && (omp_thread == 0)) {
        Print() <<"Ncells in the box = "<<  data->ncells  << "\n";
    }

    BL_PROFILE_VAR("reactor::FlatStuff", FlatStuff);
    /* Get Device MemCpy of in arrays */
    /* Get Device pointer of solution vector */
    realtype *yvec_d      = N_VGetArrayPointer(y);
    /* rhoY,T */
    std::memcpy(yvec_d, rY_in, sizeof(Real) * ((NUM_SPECIES+1)*data->ncells));
    /* rhoY_src_ext */
    std::memcpy(data->rYsrc, rY_src_in, sizeof(Real) * (NUM_SPECIES * data->ncells));
    /* rhoE/rhoH */
    std::memcpy(data->rhoX_init,   rX_in,     sizeof(Real) * data->ncells);
    std::memcpy(data->rhoXsrc_ext, rX_src_in, sizeof(Real) * data->ncells);

    /* T update with energy and Y */
    int offset;
    realtype rho, rho_inv, nrg_loc, temp;
    for  (int i = 0; i < data->ncells; i++) {
        offset = i * (NUM_SPECIES + 1);
        realtype* mass_frac = rY_in + offset;
        // get rho
        rho = 0;
        for  (int kk = 0; kk < NUM_SPECIES; kk++) {
            rho += mass_frac[kk];
        }
        rho_inv = 1 / rho;
        // get Yks
        for  (int kk = 0; kk < NUM_SPECIES; kk++) {
            mass_frac[kk] = mass_frac[kk] * rho_inv;
        }
        // get energy
        nrg_loc = rX_in[i] * rho_inv;
        // recompute T
        if (data->ireactor_type == eint_rho){
            EOS::EY2T(nrg_loc,mass_frac,temp);
        } else {
            EOS::HY2T(nrg_loc,mass_frac,temp);
        }
        // store T in y
        yvec_d[offset + NUM_SPECIES] = temp;
    }
    BL_PROFILE_VAR_STOP(FlatStuff);

    /* ReInit CVODE is faster */
    CVodeReInit(cvode_mem, time_init, y);
    
    /* There should be no internal looping of CVOde */
    data->boxcell = 0;

    BL_PROFILE_VAR("reactor::AroundCVODE", AroundCVODE);
    flag = CVode(cvode_mem, time_out, y, &dummy_time, CV_NORMAL);
    /* ONE STEP MODE FOR DEBUGGING */
    //flag = CVode(cvode_mem, time_out, y, &dummy_time, CV_ONE_STEP);
    if (check_flag(&flag, "CVode", 1)) return(1);
    BL_PROFILE_VAR_STOP(AroundCVODE);

    /* Update dt_react with real time step taken ... 
       should be very similar to input dt_react */
    dt_react = dummy_time - time_init;
#ifdef MOD_REACTOR
    /* If reactor mode is activated, update time */
    time  = time_init + dt_react;
#endif

    if ((data->iverbose > 3) && (omp_thread == 0)) {
        Print() <<"END : time curr is "<< dummy_time << " and actual dt_react is " << dt_react << "\n";
    }

    BL_PROFILE_VAR_START(FlatStuff);
    /* Pack data to return in main routine external */
    std::memcpy(rY_in, yvec_d, ((NUM_SPECIES+1)*data->ncells)*sizeof(realtype));
    for  (int i = 0; i < data->ncells; i++) {
        rX_in[i] = rX_in[i] + dt_react * rX_src_in[i];
    }

    /* T update with energy and Y */
    for  (int i = 0; i < data->ncells; i++) {
        offset = i * (NUM_SPECIES + 1);
        realtype* mass_frac = yvec_d + offset;
        // get rho
        rho = 0;
        for  (int kk = 0; kk < NUM_SPECIES; kk++) {
            rho += mass_frac[kk];
        }
        rho_inv = 1 / rho;
        // get Yks
        for  (int kk = 0; kk < NUM_SPECIES; kk++) {
            mass_frac[kk] = mass_frac[kk] * rho_inv;
        }
        // get energy
        nrg_loc = rX_in[i] * rho_inv;
        // recompute T
        if (data->ireactor_type == eint_rho){
            EOS::EY2T(nrg_loc,mass_frac,temp);
        } else {
            EOS::HY2T(nrg_loc,mass_frac,temp);
        }
        // store T in rY_in
        rY_in[offset + NUM_SPECIES] = temp;
    }
    BL_PROFILE_VAR_STOP(FlatStuff);

    if ((data->iverbose > 1) && (omp_thread == 0)) {
        Print() <<"Additional verbose info --\n";
        PrintFinalStats(cvode_mem, rY_in[NUM_SPECIES]);
        Print() <<"\n -------------------------------------\n";
    }

    /* Get estimate of how hard the integration process was */
    long int nfe,nfeLS;
    flag = CVodeGetNumRhsEvals(cvode_mem, &nfe);
    flag = CVodeGetNumLinRhsEvals(cvode_mem, &nfeLS);
    return nfe+nfeLS;
}


/* Free memory */
void ReactorCVODE_CPU::reactor_close(){

  CVodeFree(&cvode_mem);
  SUNLinSolFree(LS);

  if (data->isolve_type == dense_solve) {
      SUNMatDestroy(A);
  }

  N_VDestroy(y); 
  //FreeUserData(data);
}
/**********************************/

/**********************************/
/* Functions Called by the Solver */

/* RHS routine */
int cF_RHS(realtype t, N_Vector y_in, N_Vector ydot_in,
           void *user_data){

    realtype *y_d      = N_VGetArrayPointer(y_in);
    realtype *ydot_d   = N_VGetArrayPointer(ydot_in);

    /* Make local copies of pointers in user_data (cell M)*/
    UserData data_wk = (UserData) user_data;

    BL_PROFILE_VAR("fKernelSpec()", fKernelSpec);
    /* Loop on packed cells */
    for (int tid = 0; tid < data_wk->ncells; tid ++) {
        /* Tmp vars */
        realtype massfrac[NUM_SPECIES];
        realtype Xi[NUM_SPECIES];
        realtype cdot[NUM_SPECIES], molecular_weight[NUM_SPECIES];
        realtype cX;
        realtype temp, energy;
        realtype dt;

      /* dt is curr time - time init */
      dt = *t - time_init;

      /* Offset in case several cells */
      int offset = tid * (NUM_SPECIES + 1);

      /* MW CGS */
      CKWT(molecular_weight);

      /* rho MKS */ 
      realtype rho = 0.0;
      for (int i = 0; i < NUM_SPECIES; i++){
          rho = rho + yvec_d[offset + i];
      }

      /* temp */
      temp = yvec_d[offset + NUM_SPECIES];

      /* Yks */
      for (int i = 0; i < NUM_SPECIES; i++){
          massfrac[i] = yvec_d[offset + i] / rho;
      }

      /* NRG CGS */
      energy = (data_wk->rhoX_init[data->boxcell + tid] + data_wk->rhoXsrc_ext[data_wk->boxcell + tid] * dt) /rho;

      if (data_wk->ireactor_type == eint_rho){
          /* UV REACTOR */
          EOS::EY2T(energy, massfrac, temp);
          EOS::TY2Cv(temp, massfrac, cX);
          EOS::T2Ei(temp, Xi);
      } else if (data_wk->ireactor_type == enth_rho) {
          /* HP REACTOR */
          EOS::HY2T(energy, massfrac, temp);
          EOS::TY2Cp(temp, massfrac, cX);
          EOS::T2Hi(temp, Xi);
      }
      EOS::RTY2WDOT(rho, temp, massfrac, cdot);

      /* Fill ydot vect */
      ydot_d[offset + NUM_SPECIES] = data_wk->rhoXsrc_ext[data_wk->boxcell + tid];
      for (int i = 0; i < NUM_SPECIES; i++){
          ydot_d[offset + i] = cdot[i] + data_wk->rYsrc[(data_wk->boxcell + tid) * (NUM_SPECIES) + i];
          ydot_d[offset + NUM_SPECIES] = ydot_d[offset + NUM_SPECIES]  - ydot_d[offset + i] * Xi[i];
      }
      ydot_d[offset + NUM_SPECIES] = ydot_d[offset + NUM_SPECIES] /(rho * cX);
  }
    BL_PROFILE_VAR_STOP(fKernelSpec);

    return(0);
}
/**********************************/

/**********************************/
/* Helper functions */

/* Check function return value...
     opt == 0 means SUNDIALS function allocates memory so check if
              returned NULL pointer
     opt == 1 means SUNDIALS function returns a flag so check if
              flag >= 0
     opt == 2 means function allocates memory so check if returned
              NULL pointer */

int ReactorCVODE_CPU::check_flag(void *flagvalue, const char *funcname, int opt)
{
  int *errflag;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && flagvalue == NULL) {
      if (ParallelDescriptor::IOProcessor()) {
          fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
                  funcname);
      }
      return(1); }

  /* Check if flag < 0 */
  else if (opt == 1) {
      errflag = (int *) flagvalue;
      if (*errflag < 0) {
          if (ParallelDescriptor::IOProcessor()) {
              fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
                      funcname, *errflag);
          }
          return(1); 
      }
  }

  /* Check if function returned NULL pointer - no memory allocated */
  else if (opt == 2 && flagvalue == NULL) {
      if (ParallelDescriptor::IOProcessor()) {
          fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
                  funcname);
      }
      return(1); 
  }

  return(0);
}


/* Set or update typVals */
void ReactorCVODE_CPU::SetTypValsODE(const std::vector<double>& ExtTypVals) {
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


/* Alloc Data for CVODE */
UserData AllocUserData(int reactor_type, int num_cells)
{
  /* Make local copies of pointers in user_data */
  UserData data_wk = (UserData) malloc(sizeof *data_wk);
  int omp_thread = 0;
#ifdef _OPENMP
  omp_thread = omp_get_thread_num();
#endif

  /* ParmParse from the inputs file: only done once */
  ParmParse pp("ode");
  pp.query("analytical_jacobian",data_wk->ianalytical_jacobian);
  data_wk->iverbose = 1;
  pp.query("verbose",data_wk->iverbose);

  std::string  solve_type_str = "none";
  ParmParse ppcv("cvode");
  ppcv.query("solve_type", solve_type_str);
  /* options are: 
  dense_solve           = 1;
  sparse_solve          = 5;
  iterative_gmres_solve = 99;
  sparse_solve_custom   = 101;
  iterative_gmres_solve_custom = 199;
  hack_dump_sparsity_pattern = -5;
  */
  if (solve_type_str == "dense") {
      data_wk->isolve_type = dense_solve; 
  } else if (solve_type_str == "sparse") {
      data_wk->isolve_type = sparse_solve;
  } else if (solve_type_str == "GMRES") {
      data_wk->isolve_type = iterative_gmres_solve;
  } else if (solve_type_str == "sparse_custom") {
      data_wk->isolve_type = sparse_solve_custom;
  } else if (solve_type_str == "GMRES_custom") { 
      data_wk->isolve_type = iterative_gmres_solve_custom;
  } else if (solve_type_str == "diag") {
      data_wk->isolve_type = hack_dump_sparsity_pattern;
  } else {
      Abort("Wrong solve_type. Options are: dense, sparse, GMRES, sparse_custom, GMRES_custom");
  }

  (data_wk->ireactor_type)             = reactor_type;

  (data_wk->ncells)                    = num_cells;

  // Not sure of the interest of doing that the following: 
  //N_Vector Data = NULL; 
  //Data = N_VNew_Serial(data_wk->ncells*(NUM_SPECIES+1)); 
  //(data_wk->Yvect_full)  = N_VGetArrayPointer_Serial(Data); 
  //Data = N_VNew_Serial(data_wk->ncells*(NUM_SPECIES)); 
  //(data_wk->rYsrc)       = N_VGetArrayPointer_Serial(Data);
  //Data = N_VNew_Serial(data_wk->ncells); 
  //(data_wk->rhoX_init)   = N_VGetArrayPointer_Serial(Data);
  //Data = N_VNew_Serial(data_wk->ncells); 
  //(data_wk->rhoXsrc_ext) = N_VGetArrayPointer_Serial(Data);
   
  (data_wk->Yvect_full)  = new  amrex::Real[data_wk->ncells*(NUM_SPECIES+1)];
  (data_wk->rYsrc)       = new  amrex::Real[data_wk->ncells*(NUM_SPECIES)];
  (data_wk->rhoX_init)   = new  amrex::Real[data_wk->ncells];
  (data_wk->rhoXsrc_ext) = new  amrex::Real[data_wk->ncells];
  (data_wk->FCunt)       = new  int[data_wk->ncells];

  (data_wk->FirstTimePrecond)          = true;
  (data_wk->reactor_cvode_initialized) = false;
  (data_wk->actual_ok_to_react)        = true; 

  int HP;
  if (data_wk->ireactor_type == eint_rho) {
      HP = 0;
  } else {
      HP = 1;
  }

#ifndef USE_KLU_PP
  if (data_wk->isolve_type == iterative_gmres_solve) {
      /* Precond data */
      (data_wk->P)     = new realtype***[data_wk->ncells];
      (data_wk->Jbd)   = new realtype***[data_wk->ncells];
      (data_wk->pivot) = new sunindextype**[data_wk->ncells];
      for(int i = 0; i < data_wk->ncells; ++i) {
          (data_wk->P)[i]     = new realtype**[data_wk->ncells];
          (data_wk->Jbd)[i]   = new realtype**[data_wk->ncells];
          (data_wk->pivot)[i] = new sunindextype*[data_wk->ncells];
      }

      for(int i = 0; i < data_wk->ncells; ++i) {
          (data_wk->P)[i][i]     = newDenseMat(NUM_SPECIES+1, NUM_SPECIES+1);
          (data_wk->Jbd)[i][i]   = newDenseMat(NUM_SPECIES+1, NUM_SPECIES+1);
          (data_wk->pivot)[i][i] = newIndexArray(NUM_SPECIES+1);
      }
  //} 

#else
  /* Sparse Direct and Sparse (It) Precond data */
  data_wk->colPtrs = new int*[data_wk->ncells];
  data_wk->rowVals = new int*[data_wk->ncells];
  data_wk->Jdata   = new realtype*[data_wk->ncells];

  if (data_wk->isolve_type == sparse_solve) {
      /* Sparse Matrix for Direct Sparse KLU solver */
      (data_wk->PS) = new SUNMatrix[1];
      SPARSITY_INFO(&(data_wk->NNZ),&HP,data_wk->ncells);
      if ((data_wk->iverbose > 0) && (omp_thread == 0)) {
          Print() << "--> SPARSE solver -- non zero entries: " << data_wk->NNZ << ", which represents "<< data_wk->NNZ/float((NUM_SPECIES+1) * (NUM_SPECIES+1) * (data_wk->ncells) * (data_wk->ncells)) *100.0 <<" % fill-in pattern\n";
      }
      (data_wk->PS)[0] = SUNSparseMatrix((NUM_SPECIES+1)*data_wk->ncells, (NUM_SPECIES+1)*data_wk->ncells, data_wk->NNZ, CSC_MAT);
      data_wk->colPtrs[0] = (int*) SUNSparseMatrix_IndexPointers((data_wk->PS)[0]); 
      data_wk->rowVals[0] = (int*) SUNSparseMatrix_IndexValues((data_wk->PS)[0]);
      data_wk->Jdata[0] = SUNSparseMatrix_Data((data_wk->PS)[0]);
      SPARSITY_PREPROC_CSC(data_wk->rowVals[0],data_wk->colPtrs[0],&HP,data_wk->ncells);

  } else if (data_wk->isolve_type == iterative_gmres_solve) {
      /* KLU internal storage */
      data_wk->Common   = new klu_common[data_wk->ncells];
      data_wk->Symbolic = new klu_symbolic*[data_wk->ncells];
      data_wk->Numeric  = new klu_numeric*[data_wk->ncells];
      /* Sparse Matrices for It Sparse KLU block-solve */
      data_wk->PS = new SUNMatrix[data_wk->ncells];
      /* Number of non zero elements*/
      SPARSITY_INFO_SYST_SIMPLIFIED(&(data_wk->NNZ),&HP);
      if ((data_wk->iverbose > 0) && (omp_thread == 0) && (data_wk->ianalytical_jacobian != 0)) {
          Print() << "--> SPARSE Preconditioner -- non zero entries: " << data_wk->NNZ << ", which represents "<< data_wk->NNZ/float((NUM_SPECIES+1) * (NUM_SPECIES+1)) *100.0 <<" % fill-in pattern\n";
      }
      /* Not used yet. TODO use to fetch sparse Mat */
      data_wk->indx      = new int[data_wk->NNZ];
      data_wk->JSPSmat   = new realtype*[data_wk->ncells];
      for(int i = 0; i < data_wk->ncells; ++i) {
          (data_wk->PS)[i]    = SUNSparseMatrix(NUM_SPECIES+1, NUM_SPECIES+1, data_wk->NNZ, CSC_MAT);
          data_wk->colPtrs[i] = (int*) SUNSparseMatrix_IndexPointers((data_wk->PS)[i]); 
          data_wk->rowVals[i] = (int*) SUNSparseMatrix_IndexValues((data_wk->PS)[i]);
          data_wk->Jdata[i]   = SUNSparseMatrix_Data((data_wk->PS)[i]);
          /* indx not used YET */
          SPARSITY_PREPROC_SYST_SIMPLIFIED_CSC(data_wk->rowVals[i],data_wk->colPtrs[i],data_wk->indx,&HP);
          data_wk->JSPSmat[i] = new realtype[(NUM_SPECIES+1)*(NUM_SPECIES+1)];
          klu_defaults (&(data_wk->Common[i]));
          //data_wk->Common.btf = 0;
          //(data_wk->Common[i]).maxwork = 15;
          //data_wk->Common.ordering = 1;
          data_wk->Symbolic[i] = klu_analyze (NUM_SPECIES+1, data_wk->colPtrs[i], data_wk->rowVals[i], &(data_wk->Common[i])) ; 
      }
  //}
#endif

  } else if (data_wk->isolve_type == iterative_gmres_solve_custom) {
      /* Sparse Direct and Sparse (It) Precond data */
      data_wk->colVals = new int*[data_wk->ncells];
      data_wk->rowPtrs = new int*[data_wk->ncells];
      data_wk->Jdata   = new realtype*[data_wk->ncells];
      /* Matrices for It Sparse custom block-solve */
      data_wk->PS         = new SUNMatrix[data_wk->ncells];
      data_wk->JSPSmat    = new realtype*[data_wk->ncells];
      /* Number of non zero elements*/
      SPARSITY_INFO_SYST_SIMPLIFIED(&(data_wk->NNZ),&HP);
      for(int i = 0; i < data_wk->ncells; ++i) {
          (data_wk->PS)[i]       = SUNSparseMatrix(NUM_SPECIES+1, NUM_SPECIES+1, data_wk->NNZ, CSR_MAT);
          data_wk->rowPtrs[i]    = (int*) SUNSparseMatrix_IndexPointers((data_wk->PS)[i]);
          data_wk->colVals[i]    = (int*) SUNSparseMatrix_IndexValues((data_wk->PS)[i]);
          data_wk->Jdata[i]      = SUNSparseMatrix_Data((data_wk->PS)[i]);
          SPARSITY_PREPROC_SYST_SIMPLIFIED_CSR(data_wk->colVals[i],data_wk->rowPtrs[i],&HP,0);
          data_wk->JSPSmat[i]    = new realtype[(NUM_SPECIES+1)*(NUM_SPECIES+1)];
      }
      if ((data_wk->iverbose > 0) && (omp_thread == 0)) {
          Print() << "--> SPARSE Preconditioner -- non zero entries: " << data_wk->NNZ*data_wk->ncells << ", which represents "<< data_wk->NNZ/float((NUM_SPECIES+1) * (NUM_SPECIES+1) * data_wk->ncells) *100.0 <<" % fill-in pattern\n";
      }
  } else if (data_wk->isolve_type == sparse_solve_custom) {
      /* Number of non zero elements*/
      SPARSITY_INFO_SYST(&(data_wk->NNZ),&HP,1);
      data_wk->PSc          = SUNSparseMatrix((NUM_SPECIES+1)*data_wk->ncells, (NUM_SPECIES+1)*data_wk->ncells, data_wk->NNZ*data_wk->ncells, CSR_MAT);
      data_wk->rowPtrs_c    = (int*) SUNSparseMatrix_IndexPointers(data_wk->PSc); 
      data_wk->colVals_c    = (int*) SUNSparseMatrix_IndexValues(data_wk->PSc);
      SPARSITY_PREPROC_SYST_CSR(data_wk->colVals_c,data_wk->rowPtrs_c,&HP,data_wk->ncells,0);
      if ((data_wk->iverbose > 0) && (omp_thread == 0)) {
          Print() << "--> SPARSE solver -- non zero entries: " << data_wk->NNZ*data_wk->ncells << ", which represents "<< data_wk->NNZ/float((NUM_SPECIES+1) * (NUM_SPECIES+1) * data_wk->ncells) *100.0 <<" % fill-in pattern\n";
      }
  }  else if (data_wk->isolve_type == hack_dump_sparsity_pattern) {
      /* Debug mode, makes no sense to call with OMP/MPI activated */
      int counter;

      /* CHEMISTRY JAC */
      SPARSITY_INFO(&(data_wk->NNZ),&HP,1);
      Print() << "--> Chem Jac -- non zero entries: " << data_wk->NNZ << ", which represents "<< data_wk->NNZ/float((NUM_SPECIES+1) * (NUM_SPECIES+1)) *100.0 <<" % fill-in pattern\n";
      SUNMatrix PS;
      PS = SUNSparseMatrix((NUM_SPECIES+1), (NUM_SPECIES+1), data_wk->NNZ, CSR_MAT);
      int *colIdx, *rowCount;
      rowCount = (int*) SUNSparseMatrix_IndexPointers(PS); 
      colIdx   = (int*) SUNSparseMatrix_IndexValues(PS);
      SPARSITY_PREPROC_CSR(colIdx,rowCount,&HP,1, 0);
      std::cout <<" " << std::endl;
      std::cout << "*** Treating CHEM Jac (CSR symbolic analysis)***" << std::endl;
      std::cout <<" " << std::endl;
      int nbVals;
      counter = 0;
      for (int i = 0; i < NUM_SPECIES+1; i++) {
          nbVals         = rowCount[i+1] - rowCount[i];
          int idx_arr[nbVals];
          std::fill_n(idx_arr, nbVals, -1);
          std::memcpy(idx_arr, colIdx + rowCount[i], nbVals*sizeof(int));
          int idx        = 0;
          for (int j = 0; j < NUM_SPECIES+1; j++) {
              if ((j == idx_arr[idx]) && (nbVals > 0)) {
                  std::cout << 1 << " ";
                  idx = idx + 1;
                  counter = counter + 1;
              } else {
                  std::cout << 0 << " ";
              }
          }
          std::cout << std::endl;
      }
      std::cout << " There was " << counter << " non zero elems (compare to the "<<data_wk->NNZ<< " we need)" << std::endl;

      /* SYST JAC */
      SPARSITY_INFO_SYST(&(data_wk->NNZ),&HP,1);
      Print() << "--> Syst Jac -- non zero entries: " << data_wk->NNZ << ", which represents "<< data_wk->NNZ/float((NUM_SPECIES+1) * (NUM_SPECIES+1)) *100.0 <<" % fill-in pattern\n";
      PS = SUNSparseMatrix((NUM_SPECIES+1), (NUM_SPECIES+1), data_wk->NNZ, CSR_MAT);
      rowCount = (int*) SUNSparseMatrix_IndexPointers(PS); 
      colIdx   = (int*) SUNSparseMatrix_IndexValues(PS);
      SPARSITY_PREPROC_SYST_CSR(colIdx,rowCount,&HP,1,1);
      /* CHEMISTRY JAC */
      std::cout <<" " << std::endl;
      std::cout << "*** Treating SYST Jac (CSR symbolic analysis)***" << std::endl;
      std::cout <<" " << std::endl;
      counter = 0;
      for (int i = 0; i < NUM_SPECIES+1; i++) {
          nbVals         = rowCount[i+1] - rowCount[i];
          int idx_arr[nbVals];
          std::fill_n(idx_arr, nbVals, -1);
          std::memcpy(idx_arr, colIdx + (rowCount[i] - 1), nbVals*sizeof(int));
          int idx        = 0;
          for (int j = 0; j < NUM_SPECIES+1; j++) {
              if ((j == idx_arr[idx] - 1) && ((nbVals-idx) > 0)) {
                  std::cout << 1 << " ";
                  idx = idx + 1;
                  counter = counter + 1;
              } else {
                  std::cout << 0 << " ";
              }
          }
          std::cout << std::endl;
      }
      std::cout << " There was " << counter << " non zero elems (compare to the "<<data_wk->NNZ<< " we need)" << std::endl;

      /* SYST JAC SIMPLIFIED*/
      SPARSITY_INFO_SYST_SIMPLIFIED(&(data_wk->NNZ),&HP);
      Print() << "--> Simplified Syst Jac (for Precond) -- non zero entries: " << data_wk->NNZ << ", which represents "<< data_wk->NNZ/float((NUM_SPECIES+1) * (NUM_SPECIES+1)) *100.0 <<" % fill-in pattern\n";
      PS = SUNSparseMatrix((NUM_SPECIES+1), (NUM_SPECIES+1), data_wk->NNZ, CSR_MAT);
      rowCount = (int*) SUNSparseMatrix_IndexPointers(PS); 
      colIdx   = (int*) SUNSparseMatrix_IndexValues(PS);
      SPARSITY_PREPROC_SYST_SIMPLIFIED_CSR(colIdx,rowCount,&HP,1);
      /* CHEMISTRY JAC */
      std::cout <<" " << std::endl;
      std::cout << "*** Treating simplified SYST Jac (CSR symbolic analysis)***" << std::endl;
      std::cout <<" " << std::endl;
      counter = 0;
      for (int i = 0; i < NUM_SPECIES+1; i++) {
          nbVals         = rowCount[i+1] - rowCount[i];
          int idx_arr[nbVals];
          std::fill_n(idx_arr, nbVals, -1);
          std::memcpy(idx_arr, colIdx + (rowCount[i] - 1), nbVals*sizeof(int));
          int idx        = 0;
          for (int j = 0; j < NUM_SPECIES+1; j++) {
              if ((j == idx_arr[idx] - 1) && ((nbVals-idx) > 0)) {
                  std::cout << 1 << " ";
                  idx = idx + 1;
                  counter = counter + 1;
              } else {
                  std::cout << 0 << " ";
              }
          }
          std::cout << std::endl;
      }
      std::cout << " There was " << counter << " non zero elems (compare to the "<<data_wk->NNZ<< " we need)" << std::endl;

      Abort("Dump Sparsity Patern of different Jacobians in CSR format.");
  }

  return(data_wk);
}

/* End of file  */
