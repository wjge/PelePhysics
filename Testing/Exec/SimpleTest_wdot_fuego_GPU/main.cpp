#include <iostream>
#include <vector>

#include <AMReX_MultiFab.H>
#include <AMReX_Print.H>
#include <AMReX_VisMF.H>
#include <AMReX_ParmParse.H>
#include "mechanism.h"

#include <AMReX_GpuDevice.H>
#include <EOS.H>

using namespace amrex; 

int
main (int   argc,
      char* argv[])
{
    Initialize(argc,argv);
    {
      ParmParse pp;

      Vector<int> n_cells(BL_SPACEDIM,256);
      Box domain(IntVect(D_DECL(0,0,0)),
                 IntVect(D_DECL(n_cells[0]-1,n_cells[1]-1,n_cells[2]-1)));

      int max_size = 64;
      pp.query("max_size",max_size);

      std::string pltfile;
      bool do_plt = false;
      if (pp.countval("plotfile")>0) {
        pp.get("plotfile",pltfile);
        do_plt = true;
      }

      BoxArray ba(domain);
      ba.maxSize(max_size);

      int num_spec = NUM_SPECIES;
      int num_reac = NUM_REACTIONS;

      DistributionMapping dm(ba);

      int num_grow = 0;
      MultiFab concentrations(ba,dm,num_spec,num_grow);
      MultiFab temperature(ba,dm,1,num_grow);
      MultiFab density(ba,dm,1,num_grow);

      concentrations.setVal(0);
      temperature.setVal(0);
      density.setVal(0);

      EOS::init();

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
      {
        BL_PROFILE("INIT");
        for (MFIter mfi(concentrations,TilingIfNotGPU()); mfi.isValid(); ++mfi) {

          const Box& bx    = mfi.tilebox();

          Array4<Real> const& sc   = concentrations.array(mfi);
          Array4<Real> const& temp = temperature.array(mfi);
          Array4<Real> const& rho  = density.array(mfi);

          ParallelFor(bx,
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
            GpuArray<Real,NUM_SPECIES> Y_ker;
            GpuArray<Real,NUM_SPECIES> sc_ker;

            for (int n=0; n<num_spec; ++n) {
              Y_ker[n] = 1./num_spec;
              sc_ker[n] = 0.0;
            }

            temp(i,j,k) = 450;
            rho(i,j,k) = 0.75;

            EOS::RTY2C(rho(i,j,k), temp(i,j,k), &Y_ker[0], &sc_ker[0]);

            for (int n=0; n<num_spec; ++n) {
              sc(i,j,k,n) = sc_ker[n];
            }
          });
        }
      }

      if (do_plt) {
        MultiFab out(concentrations.boxArray(),concentrations.DistributionMap(),concentrations.nComp()+1,concentrations.nGrow());
        MultiFab::Copy(out,concentrations,0,0,concentrations.nComp(),concentrations.nGrow());
        MultiFab::Copy(out,temperature,0,concentrations.nComp(),1,concentrations.nGrow());
        std::string outfile = Concatenate(pltfile,0);
        PlotFileFromMF(concentrations,outfile);
      }

      MultiFab wdot(ba,dm,num_spec,num_grow);
      wdot.setVal(0);

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
      {
        BL_PROFILE("COMPUTE_W");
        for (MFIter mfi(concentrations,TilingIfNotGPU()); mfi.isValid(); ++mfi) {

          const Box& box = mfi.tilebox();

          Array4<Real> const& sc   = concentrations.array(mfi);
          Array4<Real> const& temp = temperature.array(mfi);
          Array4<Real> const& w    = wdot.array(mfi);

          int numPts = box.numPts();

          ParallelFor(box,
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
              productionRate(&(w(i,j,k,0)), &(sc(i,j,k,0)), temp(i,j,k)); 
          });
        }
      }
    }

    if (do_plt) {
      PlotFileFromMF(wdot,Concatenate(pltfile,1));
    }

    EOS::close();

    Finalize();

    return 0;
}

