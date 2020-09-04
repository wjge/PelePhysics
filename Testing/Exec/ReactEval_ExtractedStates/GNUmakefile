PRECISION  = DOUBLE
PROFILE    = FALSE

DEBUG      = FALSE

DIM        = 3

COMP	   = gcc
FCOMP	   = gfortran

USE_MPI    = TRUE
USE_OMP    = FALSE

TINY_PROFILE = FALSE

USE_CUDA   = FALSE

# define the location of the PELE_PHYSICS top directory
PELE_PHYSICS_HOME    ?= ../../..

#######################
# this flag activates the subcycling mode in the D/Cvode routines
DEFINES  += -DMOD_REACTOR

#######################
# ODE solver OPTIONS: DVODE (default) / SUNDIALS / RK explicit
#######################
# Activates use of SUNDIALS: CVODE (default) / ARKODE
USE_SUNDIALS_PP = TRUE
ifeq ($(USE_SUNDIALS_PP), TRUE)
  USE_ARKODE_PP = FALSE
  ifeq ($(USE_CUDA), TRUE)
      DEFINES += -DCVODE_BOXINTEG
      USE_KLU_PP = FALSE
  else
      USE_KLU_PP = TRUE
  endif
else
  # Activates use of Hari explicit RK
  # will only work if USE_SUNDIALS_PP = FALSE
  USE_RK64_PP = TRUE
endif

#######################
Eos_dir         = Fuego
Chemistry_Model = dodecane_lu
Reactions_dir   = Fuego
Transport_dir   = Simple

Bpack   := ./Make.package
Blocs   := .

include $(PELE_PHYSICS_HOME)/Testing/Exec/Make.PelePhysics
