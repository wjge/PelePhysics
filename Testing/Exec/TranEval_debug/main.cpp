#include <iostream>
#include <vector>

#include <Transport.H>
#include "mechanism.h"
#include <EOS.H>

using namespace amrex;

int
main (int   argc,
      char* argv[])
{
    Initialize(argc,argv);
    {
      EOS::init();
      transport_init();

      bool get_xi = false;
      bool get_Ddiag = true;
      bool get_lambda = true;
      bool get_mu = true;
      Real T_skin = 600.;
      Real rho_fluid = 0.57834e-3;
      Real Ddiag[NUM_SPECIES];
      Real lambda_skin = 0.;
      Real mu_skin = 0.;
      Real xi_skin = 0.;
      Real Y_skin[NUM_SPECIES]={0.0005, 0.23, 0.7695}; 

      //Set rho, set T and Y
      transport(get_xi, get_mu, get_lambda, get_Ddiag,
            T_skin, rho_fluid, Y_skin, Ddiag,
                    mu_skin, xi_skin, lambda_skin);
      //Print my xi and lambda
      for(int i=0; i<NUM_SPECIES; i++)
      {
        std::cout<<"rhoD[" <<i<<"]= "<<Ddiag[i] << 
                   ", D[" <<i<<"]= "<<Ddiag[i]/rho_fluid << std::endl;
      }
        std::cout<<"nv_skin="<<mu_skin/rho_fluid<<", lambda_skin="<<lambda_skin<<std::endl;
    }

    amrex::Finalize();

    return 0;
}
