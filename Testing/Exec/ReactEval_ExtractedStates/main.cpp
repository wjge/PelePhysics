#include <iostream>
#include <vector>
#include <list>

#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Print.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_VisMF.H>
#include <AMReX_ParmParse.H>
#include "mechanism.h"

#include <PlotFileFromMF.H>
#include <EOS.H>
#include <Transport.H>
#include <reactor.h>

#ifndef USE_RK64_PP
#ifdef USE_ARKODE_PP 
static std::string ODE_SOLVER = "ARKODE";
#else
static std::string ODE_SOLVER = "CVODE";
#endif
#else
static std::string ODE_SOLVER = "RK64";
#endif

using namespace amrex;

int
main (int   argc,
      char* argv[])
{
  Initialize(argc,argv);
  {
    BL_PROFILE_VAR("main::main()", pmain);

    ParmParse pp;

    /* ODE inputs */
    ParmParse ppode("ode");
    int ode_ncells = 1;
    ppode.query("ode_ncells",ode_ncells); // number of cells to integrate per call

    Real dt = 1.e-5;
    ppode.query("dt",dt);
    
    int ndt = 1;
    ppode.query("ndt",ndt); // number of solver calls per dt 
    
    int ode_iE = 2; // 1=full e, 2=full h
    
    Real rtol = 1e-10;
    ppode.query("rtol",rtol);
    
    Real atol = 1e-10;
    ppode.query("atol",atol);

    int use_typ_vals = 0;
    ppode.query("use_typ_vals",use_typ_vals);

    Print() << "ODE solver: " << ODE_SOLVER << std::endl;
    Print() << "Type of reactor: " << (ode_iE == 1 ? "e (PeleC)" : "h (PeleLM)") << std::endl; // <---- FIXME

    EOS::init();
    transport_init();

    BL_PROFILE_VAR("main::reactor_info()", reactInfo);

    /* Initialize reactor object inside OMP region, including tolerances */
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
      // Set ODE tolerances
      SetTolFactODE(rtol,atol);

#ifdef AMREX_USE_CUDA
      reactor_info(ode_iE, ode_ncells);
#else
      reactor_init(ode_iE, ode_ncells);
#endif
    }
    BL_PROFILE_VAR_STOP(reactInfo);

    // Read state and force data
    std::string infile;
    pp.get("infile",infile);
    std::ifstream ifs(infile.c_str());
    std::string titleLine, zoneLine;
    std::getline(ifs,titleLine);
    std::getline(ifs,zoneLine);

    // Extract variable names/counts
    auto tokens = Tokenize(titleLine," \"");
    int ist=0;
    while (tokens[ist] != "=") ist++;
    ist++;
    int Rcomp=-1, Tcomp=-1, Ycomp=-1, FYcomp=-1, FHcomp=-1, Nspec=0, NYforce=0;
    for (int i=ist; i<tokens.size(); ++i) {
      if (tokens[i]=="rho") Rcomp=i-ist;
      if (tokens[i][0]=='Y') {
        if (Ycomp<0) {
          Ycomp=i-ist;
          Nspec = 1;
        } else {
          Nspec++;
        }
      }
      if (tokens[i]=="T") Tcomp=i-ist;
      if (tokens[i]=="F_rhoh") {
        FHcomp=i-ist;
      } else if (tokens[i][0]=='F') {
        if (FYcomp<0) {
          FYcomp=i-ist;
          NYforce = 1;
        } else {
          NYforce++;
        }
      }
    }
    AMREX_ALWAYS_ASSERT(Rcomp>=0 && Rcomp+1<=tokens.size());
    AMREX_ALWAYS_ASSERT(Ycomp>=0 && Ycomp+Nspec<=tokens.size());
    AMREX_ALWAYS_ASSERT(Tcomp>=0 && Tcomp+1<=tokens.size());
    AMREX_ALWAYS_ASSERT(FYcomp>=0 && FYcomp+NYforce<=tokens.size());
    AMREX_ALWAYS_ASSERT(FHcomp>=0 && FHcomp+1<=tokens.size());
    AMREX_ALWAYS_ASSERT(Nspec==NYforce);
    AMREX_ALWAYS_ASSERT(Nspec==NUM_SPECIES);

    int Npts = -1;
    auto zoneTokens = Tokenize(zoneLine," \"");
    for (int i=0; i<tokens.size() && Npts<0; ++i) {
      auto tokenTokens = Tokenize(zoneTokens[i],"=");      
      if (tokenTokens.size() > 1) {
        if (tokenTokens[0]=="I") {
          Npts = std::stoi(tokenTokens[1]);
        }
      }
    }
    AMREX_ALWAYS_ASSERT(Npts>0);

    Box domain(IntVect(D_DECL(0,0,0)),
               IntVect(D_DECL(Npts-1,0,0)));

    FArrayBox state(domain,Nspec+2); // rhoY + rhoE + T
    FArrayBox F(domain,Nspec+1); // F[rhoY] + F[rhoH]

    std::string valLine;
    int Nvals=-1;
    for (int i=0; i<Npts; ++i) {
      std::getline(ifs,valLine);
      auto valTokens = Tokenize(valLine," ");
      if (Nvals<0) Nvals = valTokens.size();
      AMREX_ALWAYS_ASSERT(Nvals==tokens.size()-ist);
      AMREX_ALWAYS_ASSERT(valTokens.size()==Nvals);

      IntVect iv(D_DECL(i,0,0));
      Real rho, H, Y[NUM_SPECIES];
      std::stringstream(valTokens[Rcomp]) >> rho;
      for (int n=0; n<Nspec; ++n) {      
        std::stringstream(valTokens[Ycomp+n]) >> Y[n];
        std::stringstream(valTokens[FYcomp+n]) >> F(iv,n);
        state(iv,n) = rho * Y[n];
      }
      std::stringstream(valTokens[Tcomp]) >> state(iv,Nspec+1);
      std::stringstream(valTokens[FHcomp]) >> F(iv,Nspec);
      EOS::TY2H(state(iv,Nspec+1),Y,H);
      state(iv,Nspec) = rho * H;
    }
    ifs.close();

    {
      std::ofstream ofs("state_in.fab");
      state.writeOn(ofs);
      ofs.close();
    }

    Print() << "Integrating "<< domain.numPts() << " cells for: " << dt << " seconds" << std::endl;

    /* Additional defs to initialize domain */
    GpuArray<Real, AMREX_SPACEDIM>  plo = {D_DECL(0,0,0)};
    GpuArray<Real, AMREX_SPACEDIM>  dx  = {D_DECL(1,1,1)};
    GpuArray<Real, AMREX_SPACEDIM>  phi = {D_DECL(Real(domain.length(0)),
                                                  Real(domain.length(1)),
                                                  Real(domain.length(2)))};

    FArrayBox fctCount(domain,1);
    IArrayBox dummyMask(domain,1);
    dummyMask.setVal(1);

    if (use_typ_vals) {
      Print() << "Using user-defined typical values for the absolute tolerances of the ode solver.\n";
      Vector<double> typ_vals(NUM_SPECIES+1);
      ppode.getarr("typ_vals", typ_vals,0,NUM_SPECIES+1);
      for (int i = 0; i < NUM_SPECIES; ++i) {
        typ_vals[i] = std::max(typ_vals[i],1.e-10);
      }
      SetTypValsODE(typ_vals);
    }

    /* REACT */
    BL_PROFILE_VAR("Advance",Advance);
    BL_PROFILE_VAR_NS("React",ReactInLoop);
    BL_PROFILE_VAR_NS("Allocs",Allocs);
    BL_PROFILE_VAR_NS("Flatten",mainflatten);

    int nc          = domain.numPts();
    int extra_cells = 0;

    const auto len     = length(domain);
    const auto lo      = lbound(domain);

    auto const& rhoY    = state.array();
    auto const& T       = state.array(NUM_SPECIES+1);
    auto const& rhoH    = state.array(NUM_SPECIES);
    auto const& frcExt  = F.array();
    auto const& frcEExt = F.array(NUM_SPECIES);
    auto const& fc      = fctCount.array();
    auto const& mask    = dummyMask.array();

#ifdef AMREX_USE_CUDA
    cudaError_t cuda_status = cudaSuccess;
    ode_ncells    = nc;
#else
#ifndef CVODE_BOXINTEG
    extra_cells = nc - (nc / ode_ncells) * ode_ncells; 
#endif
#endif

    Print() << " Integrating " << nc << " cells with a "<<ode_ncells<< " ode cell buffer \n";
    Print() << "("<< extra_cells<<" extra cells) \n";

#ifndef CVODE_BOXINTEG
    BL_PROFILE_VAR_START(Allocs);
    int nCells               =  nc+extra_cells;
    auto tmp_vect            =  new Real[nCells * (NUM_SPECIES+1)];
    auto tmp_src_vect        =  new Real[nCells * NUM_SPECIES];
    auto tmp_vect_energy     =  new Real[nCells];
    auto tmp_src_vect_energy =  new Real[nCells];
    auto tmp_fc              =  new Real[nCells];
    auto tmp_mask            =  new int[nCells];
    BL_PROFILE_VAR_STOP(Allocs);

    BL_PROFILE_VAR_START(mainflatten);
    ParallelFor(domain,
    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
    {
      int icell = (k-lo.z)*len.x*len.y + (j-lo.y)*len.x + (i-lo.x);
      for(int sp=0; sp<NUM_SPECIES; sp++) {
        tmp_vect[icell*(NUM_SPECIES+1)+sp]     = rhoY(i,j,k,sp);
        tmp_src_vect[icell*NUM_SPECIES+sp]     = frcExt(i,j,k,sp);
      }
      tmp_vect[icell*(NUM_SPECIES+1)+NUM_SPECIES] = T(i,j,k);
      tmp_vect_energy[icell]                      = rhoH(i,j,k);
      tmp_src_vect_energy[icell]                  = frcEExt(i,j,k);
      tmp_mask[icell]                             = mask(i,j,k);
    });

    for (int icell=nc; icell<nc+extra_cells; icell++) {
      for(int sp=0; sp<NUM_SPECIES; sp++) {
        tmp_vect[icell*(NUM_SPECIES+1)+sp]     = rhoY(0,0,0,sp);
        tmp_src_vect[icell*NUM_SPECIES+sp]     = frcExt(0,0,0,sp);
      }
      tmp_vect[icell*(NUM_SPECIES+1)+NUM_SPECIES] = T(0,0,0);
      tmp_vect_energy[icell]                      = rhoH(0,0,0); 
      tmp_src_vect_energy[icell]                  = frcEExt(0,0,0);
      tmp_mask[icell]                             = mask(0,0,0);
    }
    BL_PROFILE_VAR_STOP(mainflatten);
#endif

      /* Solve */
    BL_PROFILE_VAR_START(ReactInLoop);
#ifndef CVODE_BOXINTEG
    for(int i = 0; i < nCells; i+=ode_ncells) {
      if (tmp_mask[i]==1)
      {
        tmp_fc[i] = 0.0;
        Real time = 0.0;
        Real dt_incr = dt/ndt;
        for (int ii = 0; ii < ndt; ++ii) {
          tmp_fc[i] += react(tmp_vect + i*(NUM_SPECIES+1), tmp_src_vect + i*NUM_SPECIES,
                             tmp_vect_energy + i, tmp_src_vect_energy + i,
                             dt_incr, time);
          dt_incr =  dt/ndt;
        }
      }
    }
#else
    {      
      Real time = 0.0;
      Real dt_incr = dt/ndt;
      for (int ii = 0; ii < ndt; ++ii)
      {
#ifdef AMREX_USE_CUDA
        react(box,
              rhoY, frcExt, T,
              rhoH, frcEExt,
              fc, mask,
              dt_incr, time,
              ode_iE, Gpu::gpuStream());
#else
        react(box,
              rhoY, frcExt, T,
              rhoH, frcEExt,
              fc, mask,
              dt_incr, time);
#endif
        dt_incr =  dt/ndt;
      }
    }
#endif
    BL_PROFILE_VAR_STOP(ReactInLoop);
      
#ifndef CVODE_BOXINTEG
    BL_PROFILE_VAR_START(mainflatten);
    ParallelFor(domain,
    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
    {
      int icell = (k-lo.z)*len.x*len.y + (j-lo.y)*len.x + (i-lo.x);
      for(int sp=0; sp<NUM_SPECIES; sp++) {
        rhoY(i,j,k,sp)        = tmp_vect[icell*(NUM_SPECIES+1)+sp];
      }
      T(i,j,k)                = tmp_vect[icell*(NUM_SPECIES+1) + NUM_SPECIES];
      rhoH(i,j,k)             = tmp_vect_energy[icell];
      fc(i,j,k)               = tmp_fc[icell];
    });
    BL_PROFILE_VAR_STOP(mainflatten);

    delete[] tmp_vect;
    delete[] tmp_src_vect;
    delete[] tmp_vect_energy;
    delete[] tmp_src_vect_energy;
    delete[] tmp_fc;
    delete[] tmp_mask;
#endif
    BL_PROFILE_VAR_STOP(Advance);

    {
      std::ofstream ofs("state_out.fab");
      state.writeOn(ofs);
      ofs.close();
    }

    {
      Vector<double> typ_vals(NUM_SPECIES+1);
      Print() << "ode.typ_vals= ";
      for (int i = 0; i < NUM_SPECIES+1; ++i) {
        Print() << std::max(1.e-10,state.max(i)) << " ";
      }
      Print() << std::endl;

      Print() << "Trange = " << state.min(NUM_SPECIES+1) << " : " << state.max(NUM_SPECIES+1) << std::endl;
    }
    
#ifndef AMREX_USE_CUDA
    reactor_close();
#endif
    transport_close();
    EOS::close();
    
    BL_PROFILE_VAR_STOP(pmain);
  }
  Finalize();

  return 0;
}
