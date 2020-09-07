PRECISION  = DOUBLE
PROFILE    = FALSE

DEBUG      = TRUE

DIM        = 2

COMP	   = gcc
FCOMP	   = gfortran

USE_MPI    = FALSE
USE_OMP    = FALSE

FUEGO_GAS  = TRUE

# define the location of the PELE_PHYSICS top directory
PELE_PHYSICS_HOME    := ../../..

TRANSPORT_CPP = TRUE

  Eos_dir         = Fuego
  Chemistry_Model = decane_3sp
  Reactions_dir   = Fuego
  Transport_dir   = Simple

Bpack   := ./Make.package
Blocs   := .

include $(PELE_PHYSICS_HOME)/Testing/Exec/Make.PelePhysics
