import numpy as np
import sys
import os

mech = sys.argv[2]
mech_file = os.environ['PELE_PHYSICS_HOME'] + '/Support/Fuego/Mechanism/Models/' + mech + '/mechanism.h'
print('grabbing species list from ' + mech_file)
f = open(mech_file)
lines = f.readlines()
f.close()

species = []
for l in lines:
    tok = l.split()
    if len(tok)==3 and '_ID' in tok[1]:
        species.append(tok[1][:-3])
nspec = len(species)
print('Found '+str(nspec)+' species names to use')

f = open(sys.argv[1])
lines = f.readlines()
f.close()

d = np.loadtxt(sys.argv[1])
npts = d.shape[0]
ns = d.shape[1] # (u,v,w,rho,rhoY1...rhoYN,rhoh,t,Temp,FrhoY[1-N],Frhoh) = 9+nspec+nspec
print('Number of states = '+str(ns)+' [expecting '+str(9+2*nspec)+']')

if len(species) + 8 + len(species) + 1 != ns:
    sys.exit('Number of species not compatible with sample file')

f = open(sys.argv[3],'w')
f.write('VARIABLES = "rho" "T" ' + ' '.join(['"Y_'+x+'"' for x in species]) + ' ' + ' '.join(['"F_'+x+'"' for x in species])+' "F_rhoh"\n')
f.write('ZONE I=' + str(npts) + ' FORMAT=POINT\n')
si = np.argsort(d[:,nspec+6])
#for i in range(npts):
for i in si:
    rho = np.sum(d[i,4:4+nspec])
    Y = [x/rho for x in d[i,4:4+nspec]]
    T = d[i,nspec+6]
    F = d[i,nspec+8:]
    f.write(str(rho) + ' ' + str(T) +' '+ ' '.join([str(Y[j]) for j in range(nspec)])+' '+' '.join([str(F[j]) for j in range(nspec+1)])+'\n')
f.close()

# Check that file makes sense
f = open(sys.argv[3])
lines = f.readlines()
f.close()

ntot = len(lines[2].strip().split())

for line in lines[3:]:
    if ntot != len(line.strip().split()):
        print('n ne ntot '+sstr(n)+', '+str(ntot))
        sys.exit()
