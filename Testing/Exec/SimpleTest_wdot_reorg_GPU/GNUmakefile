PRECISION  = DOUBLE
PROFILE    = FALSE

DEBUG      = FALSE

DIM        = 3

COMP	   = gcc

USE_MPI    = FALSE
USE_OMP    = FALSE

FUEGO_GAS  = TRUE

TINY_PROFILE = TRUE

USE_CUDA   = TRUE

# define the location of the PELE_PHYSICS top directory
PELE_PHYSICS_HOME    ?= ../../..

ifeq ($(FUEGO_GAS), TRUE)
  Eos_dir         = Fuego
  Chemistry_Model = drm19
  Reactions_dir   = Fuego
  Transport_dir   = Simple
else
  Chemistry_Model = Null
  Eos_dir       = GammaLaw
  Reactions_dir = Fuego
  Transport_dir = Constant
endif

Bpack   := ./Make.package
Blocs   := .

include $(PELE_PHYSICS_HOME)/Testing/Exec/Make.PelePhysics
