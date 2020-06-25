#include <math.h>
#include <iostream>
#include <cstring>
#include <chrono>
#include <cassert>
#include <assert.h>

#include <arkode/arkode_arkstep.h>
#include <arkode/arkode_erkstep.h>

#include <nvector/nvector_cuda.h>
#include <nvector/nvector_serial.h>    /* access to serial N_Vector            */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <cuda_runtime_api.h>

#include <AMReX_Print.H>
/**********************************/

namespace reactor_arrays
{
    extern realtype *rhoe_init;
    extern realtype *rhoesrc_ext;
    extern realtype *rYsrc;
    extern N_Vector y;

    void allocate_reactor_vecs(int ncells,cudaStream_t stream);
    void deallocate_reactor_vecs();
    realtype* get_device_pointer();
}

typedef struct ARKODEUserData {
    /* Checks */
    bool reactor_arkode_initialized;
    /* Base items */
    int ncells_d[1];
    int neqs_per_cell[1];
    int iverbose;
    int ireactor_type;
    double dt_save;

    cudaStream_t stream;
    int nbBlocks;
    int nbThreads;
} *UserData;

int reactor_info(const int* cvode_iE, const int* Ncells);

static int cF_RHS(realtype t, N_Vector y_in, N_Vector ydot, void *user_data);

void reactor_close();
    
int react( realtype *dt_react, realtype *time,
           const int* cvode_iE, const int* Ncells, 
           cudaStream_t stream,double reltol=1e-6,double abstol=1e-10);


AMREX_GPU_DEVICE
inline
void
fKernelSpec(int ncells, void *user_data, 
            realtype *yvec_d, realtype *ydot_d,  
            double *rhoX_init, double *rhoXsrc_ext, double *rYs);
