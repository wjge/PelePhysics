#!/usr/bin/env python
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2003 All Rights Reserved
# 
#  <LicenseText>
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
from collections import defaultdict, OrderedDict

from weaver.mills.CMill import CMill

from pyre.units.pressure import atm
from pyre.units.SI import meter, second, mole, kelvin
from pyre.units.length import cm
from pyre.units.energy import cal, kcal, J, kJ, erg
from pyre.handbook.constants.fundamental import boltzmann
from pyre.handbook.constants.fundamental import avogadro
from pyre.handbook.constants.fundamental import gas_constant as R

import sys
import numpy as np

smallnum = 1e-100
R = 8.31446261815324e7 * erg/mole/kelvin
Rc = 1.98721558317399615845 * cal/mole/kelvin
Patm = 1013250.0
sym  = ""
fsym = "_"

class speciesDb:
    def __init__(self, id, name, mwt):
        self.id = id
        self.symbol = name
        self.weight = mwt
        return


class CPickler(CMill):

    def __init__(self):
        CMill.__init__(self)
        self.species = []
        self.qss_species = []
        self.qss_list = []
        self.nSpecies = 0
        self.nQSSspecies = 0
        self.reactionIndex = []
        self.lowT = 100.0
        self.highT = 10000.0
        # idk if these should be put elsewhere (or if this is even the best way to do this)
        self.QSS_mech_index = {}
        return


    def _setSpecies(self, mechanism):
        """ For internal use """
        import pyre
        periodic = pyre.handbook.periodicTable()
 
        nSpecies    = len(mechanism.species())
        nQSSspecies = len(mechanism.qss_species())

        print nSpecies
        print nQSSspecies
        print

        self.nSpecies = nSpecies
        self.nQSSspecies = nQSSspecies
        
        # Split into two loops eventually for mech.species and mech.qss_species
        # Currently just minorly manipulated to split species from qss_species b/c right now mech.species also contains the QSS species
        
        for qss_sp in mechanism.qss_species():
            weight = 0.0
            for elem, coef in qss_sp.composition:
                aw = mechanism.element(elem).weight
                if not aw:
                    aw = periodic.symbol(elem.capitalize()).atomicWeight
                weight += coef * aw

            tempspqss = speciesDb(qss_sp.id, qss_sp.symbol, weight)
            self.qss_species.append(tempspqss)
            self.qss_list.append(qss_sp.symbol)

        nonQSS = []
        for species in mechanism.species():
            if species.symbol not in self.qss_list:
                weight = 0.0 
                for elem, coef in species.composition:
                    aw = mechanism.element(elem).weight
                    if not aw:
                        aw = periodic.symbol(elem.capitalize()).atomicWeight
                    weight += coef * aw

                tempsp = speciesDb(species.id, species.symbol, weight)
                self.species.append(tempsp)
                nonQSS.append(species.symbol)

        # Initialize QSS species-species, species-reaction, and species coupling networks        
        self.QSS_SSnet = np.zeros([self.nQSSspecies, self.nQSSspecies], 'd')
        self.QSS_SRnet = np.zeros([self.nQSSspecies, len(mechanism.reaction())], 'd')
        self.QSS_SCnet = np.zeros([self.nQSSspecies, self.nQSSspecies], 'd')

        print "Transported species list =", list(nonQSS)
        print "QSS species list =", self.qss_list

        return


    ##########################
    #This is the main simplified routine for QSS
    #called in weaver/weaver/mills/Mill.py
    #Eventually we will have to add all missing pieces back
    ##########################
    def _renderDocument_QSS(self, mechanism, options=None):

        print "we are now in CPickler !"

        self._setSpecies(mechanism)
        self.reactionIndex = mechanism._sort_reactions()

        # self._test2(mechanism)
            
        #chemistry_file.H
        self._includes(True)
        self._header_QSS(mechanism)
        self._namespace(mechanism)
        #chemistry_file.H

        self._includes_chop()
        self._statics_QSS(mechanism)
        self._ckinit_QSS(mechanism)

        # chemkin wrappers
        self._ckindx(mechanism)
        self._cksyme_str(mechanism)
        self._cksyms_str(mechanism)
        self._ckrp(mechanism)
        self._ckpx(mechanism)
        self._ckpy(mechanism)
        self._ckpc(mechanism)
        self._ckrhox(mechanism)
        self._ckrhoy(mechanism)
        self._ckrhoc(mechanism)
        self._ckwt(mechanism)
        self._ckawt(mechanism)
        self._ckmmwy(mechanism)
        self._ckmmwx(mechanism)
        self._ckmmwc(mechanism)
        self._ckytx(mechanism)
        self._ckytcp(mechanism)
        self._ckytcr(mechanism)
        self._ckxty(mechanism)
        self._ckxtcp(mechanism)
        self._ckxtcr(mechanism)
        self._ckctx(mechanism)
        self._ckcty(mechanism)
        # polyfits 
        self._ckcpor(mechanism)
        self._ckhort(mechanism)
        self._cksor(mechanism)
        
        self._ckcvms(mechanism)
        self._ckcpms(mechanism)
        self._ckums(mechanism)
        self._ckhms(mechanism)
        self._ckgms(mechanism)
        self._ckams(mechanism)
        self._cksms(mechanism)

        self._ckcpbl(mechanism)
        self._ckcpbs(mechanism)
        self._ckcvbl(mechanism)
        self._ckcvbs(mechanism)
        
        self._ckhbml(mechanism)
        self._ckhbms(mechanism)
        self._ckubml(mechanism)
        self._ckubms(mechanism)
        self._cksbml(mechanism)
        self._cksbms(mechanism)
        self._ckgbml(mechanism)
        self._ckgbms(mechanism)
        self._ckabml(mechanism)
        self._ckabms(mechanism)

        self._ckwc(mechanism)
        self._ckwyp(mechanism)
        self._ckwxp(mechanism)
        self._ckwyr(mechanism)
        self._ckwxr(mechanism)
        
        self._ckncf(mechanism)
        
        # Fuego Functions
        # ORI CPU version
        self._productionRate(mechanism)
        # should be chopped
        self._test(mechanism)
        self._QSScomponentFunctions(mechanism)
        # ORI CPU vectorized version
        self._DproductionRatePrecond(mechanism)
        self._DproductionRate(mechanism)
        self._sparsity(mechanism)
        # ORI CPU version
        # For now, let's ignore jacobians... BUT we need a dummy routine for PELE
        self._ajac_dummy(mechanism)
        self._ajacPrecond_dummy(mechanism)
        self._dthermodT(mechanism)
        self._equilibriumConstants(mechanism)
        self._thermo_GPU(mechanism)
        self._atomicWeight(mechanism)
        self._T_given_ey(mechanism)
        self._T_given_hy(mechanism)
        #AF: add transport data
        self._transport(mechanism)
        #AF: dummy gjs routines
        self._emptygjs(mechanism)

        ### MECH HEADER -- second file starts here
        self._print_mech_header(mechanism)
        ### MECH HEADER

        return


    #Pieces for the file chemistry_file.H#
    def _includes(self, header):
        self._rep += [
            '#include <math.h>',
            '#include <stdio.h>',
            '#include <string.h>'
        ]
        if header:
            self._rep += [
                '#include <stdlib.h>',
                '#include <vector>',
                '#include <AMReX_Gpu.H>'
            ]
        else:
            self._rep += [
                '#include <stdlib.h>'
            ]

        return


    def _header_QSS(self, mechanism):
        self._rep += [
            '',
            'extern "C"',
            '{',
            'AMREX_GPU_HOST_DEVICE void get_imw(double imw_new[]);',
            'AMREX_GPU_HOST_DEVICE void get_mw(double mw_new[]);',
            'void atomicWeight(double *  awt);',
            'void molecularWeight(double *  wt);',
            'AMREX_GPU_HOST_DEVICE void gibbs(double *  species, double *  tc);',
            'AMREX_GPU_HOST_DEVICE void gibbs_qss(double *  species, double *  tc);',
            'AMREX_GPU_HOST_DEVICE void helmholtz(double *  species, double *  tc);',
            'AMREX_GPU_HOST_DEVICE void speciesInternalEnergy(double *  species, double *  tc);',
            'AMREX_GPU_HOST_DEVICE void speciesEnthalpy(double *  species, double *  tc);',
            'AMREX_GPU_HOST_DEVICE void speciesEntropy(double *  species, double *  tc);',
            'AMREX_GPU_HOST_DEVICE void cp_R(double *  species, double *  tc);',
            'AMREX_GPU_HOST_DEVICE void cv_R(double *  species, double *  tc);',
            'void equilibriumConstants(double *  kc, double *  g_RT, double T);',
            'AMREX_GPU_HOST_DEVICE void productionRate(double *  wdot, double *  sc, double T);',
            'AMREX_GPU_HOST_DEVICE void comp_qfqr(double *  q_f, double *  q_r, double *  sc, double *  tc, double invT);',
            '#ifndef AMREX_USE_CUDA',
            'void comp_k_f(double *  tc, double invT, double *  k_f);',
            'void comp_Kc(double *  tc, double invT, double *  Kc);',
            '#endif',
            'AMREX_GPU_HOST_DEVICE void CKINIT'+sym+'();',
            'AMREX_GPU_HOST_DEVICE void CKFINALIZE'+sym+'();',
            'void CKINDX'+sym+'(int * mm, int * kk, int * ii, int * nfit );',
            'void CKSYME_STR(amrex::Vector<std::string>& ename);',
            'void CKSYMS_STR(amrex::Vector<std::string>& kname);',
            'void CKRP'+sym+'(double *  ru, double *  ruc, double *  pa);',
            'void CKPX'+sym+'(double *  rho, double *  T, double *  x, double *  P);',
            'AMREX_GPU_HOST_DEVICE void CKPY'+sym+'(double *  rho, double *  T, double *  y, double *  P);',
            'void CKPC'+sym+'(double *  rho, double *  T, double *  c, double *  P);',
            'void CKRHOX'+sym+'(double *  P, double *  T, double *  x, double *  rho);',
            'AMREX_GPU_HOST_DEVICE void CKRHOY'+sym+'(double *  P, double *  T, double *  y, double *  rho);',
            'void CKRHOC'+sym+'(double *  P, double *  T, double *  c, double *  rho);',
            'void CKWT'+sym+'(double *  wt);',
            'void CKAWT'+sym+'(double *  awt);',
            'AMREX_GPU_HOST_DEVICE void CKMMWY'+sym+'(double *  y, double *  wtm);',
            'void CKMMWX'+sym+'(double *  x, double *  wtm);',
            'void CKMMWC'+sym+'(double *  c, double *  wtm);',
            'AMREX_GPU_HOST_DEVICE void CKYTX'+sym+'(double *  y, double *  x);',
            'void CKYTCP'+sym+'(double *  P, double *  T, double *  y, double *  c);',
            'AMREX_GPU_HOST_DEVICE void CKYTCR'+sym+'(double *  rho, double *  T, double *  y, double *  c);',
            'AMREX_GPU_HOST_DEVICE void CKXTY'+sym+'(double *  x, double *  y);',
            'void CKXTCP'+sym+'(double *  P, double *  T, double *  x, double *  c);',
            'void CKXTCR'+sym+'(double *  rho, double *  T, double *  x, double *  c);',
            'void CKCTX'+sym+'(double *  c, double *  x);',
            'void CKCTY'+sym+'(double *  c, double *  y);',
            'void CKCPOR'+sym+'(double *  T, double *  cpor);',
            'void CKHORT'+sym+'(double *  T, double *  hort);',
            'void CKSOR'+sym+'(double *  T, double *  sor);',
            
            'AMREX_GPU_HOST_DEVICE void CKCVMS'+sym+'(double *  T, double *  cvms);',
            'AMREX_GPU_HOST_DEVICE void CKCPMS'+sym+'(double *  T, double *  cvms);',
            'AMREX_GPU_HOST_DEVICE void CKUMS'+sym+'(double *  T, double *  ums);',
            'AMREX_GPU_HOST_DEVICE void CKHMS'+sym+'(double *  T, double *  ums);',
            'void CKGMS'+sym+'(double *  T, double *  gms);',
            'void CKAMS'+sym+'(double *  T, double *  ams);',
            'void CKSMS'+sym+'(double *  T, double *  sms);',
            
            'void CKCPBL'+sym+'(double *  T, double *  x, double *  cpbl);',
            'AMREX_GPU_HOST_DEVICE void CKCPBS'+sym+'(double *  T, double *  y, double *  cpbs);',
            'void CKCVBL'+sym+'(double *  T, double *  x, double *  cpbl);',
            'AMREX_GPU_HOST_DEVICE void CKCVBS'+sym+'(double *  T, double *  y, double *  cpbs);',
            
            'void CKHBML'+sym+'(double *  T, double *  x, double *  hbml);',
            'AMREX_GPU_HOST_DEVICE void CKHBMS'+sym+'(double *  T, double *  y, double *  hbms);',
            'void CKUBML'+sym+'(double *  T, double *  x, double *  ubml);',
            'AMREX_GPU_HOST_DEVICE void CKUBMS'+sym+'(double *  T, double *  y, double *  ubms);',
            'void CKSBML'+sym+'(double *  P, double *  T, double *  x, double *  sbml);',
            'void CKSBMS'+sym+'(double *  P, double *  T, double *  y, double *  sbms);',
            'void CKGBML'+sym+'(double *  P, double *  T, double *  x, double *  gbml);',
            'void CKGBMS'+sym+'(double *  P, double *  T, double *  y, double *  gbms);',
            'void CKABML'+sym+'(double *  P, double *  T, double *  x, double *  abml);',
            'void CKABMS'+sym+'(double *  P, double *  T, double *  y, double *  abms);',

            
            'AMREX_GPU_HOST_DEVICE void CKWC'+sym+'(double *  T, double *  C, double *  wdot);',
            'void CKWYP'+sym+'(double *  P, double *  T, double *  y, double *  wdot);',
            'void CKWXP'+sym+'(double *  P, double *  T, double *  x, double *  wdot);',
            'AMREX_GPU_HOST_DEVICE void CKWYR'+sym+'(double *  rho, double *  T, double *  y, double *  wdot);',
            'void CKWXR'+sym+'(double *  rho, double *  T, double *  x, double *  wdot);',

            'void CKNCF'+sym+'(int * ncf);',
            
            'AMREX_GPU_HOST_DEVICE void DWDOT(double *  J, double *  sc, double *  T, int * consP);',
            'AMREX_GPU_HOST_DEVICE void DWDOT_SIMPLIFIED(double *  J, double *  sc, double *  Tp, int * HP);',
            'AMREX_GPU_HOST_DEVICE void SPARSITY_INFO(int * nJdata, int * consP, int NCELLS);',
            'AMREX_GPU_HOST_DEVICE void SPARSITY_INFO_SYST(int * nJdata, int * consP, int NCELLS);',
            'AMREX_GPU_HOST_DEVICE void SPARSITY_INFO_SYST_SIMPLIFIED(int * nJdata, int * consP);',
            'AMREX_GPU_HOST_DEVICE void SPARSITY_PREPROC_CSC(int * rowVals, int * colPtrs, int * consP, int NCELLS);',
            'AMREX_GPU_HOST_DEVICE void SPARSITY_PREPROC_CSR(int * colVals, int * rowPtrs, int * consP, int NCELLS, int base);',
            'AMREX_GPU_HOST_DEVICE void SPARSITY_PREPROC_SYST_CSR(int * colVals, int * rowPtrs, int * consP, int NCELLS, int base);',
            'AMREX_GPU_HOST_DEVICE void SPARSITY_PREPROC_SYST_SIMPLIFIED_CSC(int * rowVals, int * colPtrs, int * indx, int * consP);',
            'AMREX_GPU_HOST_DEVICE void SPARSITY_PREPROC_SYST_SIMPLIFIED_CSR(int * colVals, int * rowPtr, int * consP, int base);',
            'AMREX_GPU_HOST_DEVICE void aJacobian(double *  J, double *  sc, double T, int consP);',
            'AMREX_GPU_HOST_DEVICE void aJacobian_precond(double *  J, double *  sc, double T, int HP);',
            'AMREX_GPU_HOST_DEVICE void dcvpRdT(double *  species, double *  tc);',
            'AMREX_GPU_HOST_DEVICE void GET_T_GIVEN_EY(double *  e, double *  y, double *  t, int *ierr);',
            'AMREX_GPU_HOST_DEVICE void GET_T_GIVEN_HY(double *  h, double *  y, double *  t, int *ierr);',
             self.line('Transport function declarations'),
             'void egtransetLENIMC(int* LENIMC);',
             'void egtransetLENRMC(int* LENRMC);',
             'void egtransetNO(int* NO);',
             'void egtransetKK(int* KK);',
             'void egtransetNLITE(int* NLITE);',
             'void egtransetPATM(double* PATM);',
             'void egtransetWT(double* WT);',
             'void egtransetEPS(double* EPS);',
             'void egtransetSIG(double* SIG);',
             'void egtransetDIP(double* DIP);',
             'void egtransetPOL(double* POL);',
             'void egtransetZROT(double* ZROT);',
             'void egtransetNLIN(int* NLIN);',
             'void egtransetCOFETA(double* COFETA);',
             'void egtransetCOFLAM(double* COFLAM);',
             'void egtransetCOFD(double* COFD);',
             'void egtransetKTDIF(int* KTDIF);',
             '/* gauss-jordan solver external routine*/',
             'AMREX_GPU_HOST_DEVICE void sgjsolve(double* A, double* x, double* b);',
             'AMREX_GPU_HOST_DEVICE void sgjsolve_simplified(double* A, double* x, double* b);',
             '}',
             ]

        return


    def _namespace(self,mechanism):
        self._write()
        self._write('#ifndef AMREX_USE_CUDA')
        self._write('namespace thermo')
        self._write('{')

        self._indent()

        nReactions = len(mechanism.reaction())
        self._write()
        self._write('extern double fwd_A[%d], fwd_beta[%d], fwd_Ea[%d];' 
                    % (nReactions,nReactions,nReactions))
        self._write('extern double low_A[%d], low_beta[%d], low_Ea[%d];' 
                    % (nReactions,nReactions,nReactions))
        self._write('extern double rev_A[%d], rev_beta[%d], rev_Ea[%d];' 
                    % (nReactions,nReactions,nReactions))
        self._write('extern double troe_a[%d],troe_Ts[%d], troe_Tss[%d], troe_Tsss[%d];' 
                    % (nReactions,nReactions,nReactions,nReactions))
        self._write('extern double sri_a[%d], sri_b[%d], sri_c[%d], sri_d[%d], sri_e[%d];'
                    % (nReactions,nReactions,nReactions,nReactions,nReactions))
        self._write('extern double activation_units[%d], prefactor_units[%d], phase_units[%d];'
                    % (nReactions,nReactions,nReactions))
        self._write('extern int is_PD[%d], troe_len[%d], sri_len[%d], nTB[%d], *TBid[%d];' 
                    % (nReactions,nReactions,nReactions,nReactions,nReactions))
        self._write('extern double *TB[%d];' 
                    % (nReactions))

        self._write('extern std::vector<std::vector<double>> kiv; ')
        self._write('extern std::vector<std::vector<double>> nuv; ')
        self._write('extern std::vector<std::vector<double>> kiv_qss; ')
        self._write('extern std::vector<std::vector<double>> nuv_qss; ')

        self._outdent()

        self._write('}')
        self._write('#endif')

        return
    #Pieces for the file chemistry_file.H#



    #Pieces for mechanism.cpp
    def _includes_chop(self):
        self._rep += [
            '#include "chemistry_file.H"'
            ]
        return


    def _statics_QSS(self,mechanism):
        nReactions = len(mechanism.reaction())
        nSpecies = len(mechanism.species())

        ispecial   = self.reactionIndex[5:7]

        nspecial   = ispecial[1]   - ispecial[0]

        self._write()

        self._write('#ifndef AMREX_USE_CUDA')
        self._write('namespace thermo')
        self._write('{')
        self._indent()
        self._write('double fwd_A[%d], fwd_beta[%d], fwd_Ea[%d];' 
                    % (nReactions,nReactions,nReactions))
        self._write('double low_A[%d], low_beta[%d], low_Ea[%d];' 
                    % (nReactions,nReactions,nReactions))
        self._write('double rev_A[%d], rev_beta[%d], rev_Ea[%d];' 
                    % (nReactions,nReactions,nReactions))
        self._write('double troe_a[%d],troe_Ts[%d], troe_Tss[%d], troe_Tsss[%d];' 
                    % (nReactions,nReactions,nReactions,nReactions))
        self._write('double sri_a[%d], sri_b[%d], sri_c[%d], sri_d[%d], sri_e[%d];'
                    % (nReactions,nReactions,nReactions,nReactions,nReactions))
        self._write('double activation_units[%d], prefactor_units[%d], phase_units[%d];'
                    % (nReactions,nReactions,nReactions))
        self._write('int is_PD[%d], troe_len[%d], sri_len[%d], nTB[%d], *TBid[%d];' 
                    % (nReactions,nReactions,nReactions,nReactions,nReactions))
        self._write('double *TB[%d];' 
                    % (nReactions))

        if nspecial > 0:  
                self._write('double prefactor_units_rev[%d], activation_units_rev[%d];' 
                            % (nReactions,nReactions))

        self._write('std::vector<std::vector<double>> kiv(%d); ' % (nReactions))
        self._write('std::vector<std::vector<double>> nuv(%d); ' % (nReactions))
        self._write('std::vector<std::vector<double>> kiv_qss(%d); ' % (nReactions))
        self._write('std::vector<std::vector<double>> nuv_qss(%d); ' % (nReactions))

        self._outdent()

        self._write('};')

        self._write()

        self._write('using namespace thermo;')
        self._write('#endif')
        self._write()


        self._write(self.line(' Inverse molecular weights'))
        self._write('static AMREX_GPU_DEVICE_MANAGED double imw[%d] = {' %nSpecies )
        self._indent()
        for i in range(0,self.nSpecies):
            species = self.species[i]
            text = '1.0 / %f' % (species.weight)
            if (i<self.nSpecies-1):
               text += ',  '
            else:
               text += '};  '
            self._write(text + self.line('%s' % species.symbol))
        self._outdent()
        self._write()

        self._write(self.line(' Inverse molecular weights'))
        self._write('static AMREX_GPU_DEVICE_MANAGED double molecular_weights[%d] = {' %nSpecies )
        self._indent()
        for i in range(0,self.nSpecies):
            species = self.species[i]
            text = '%f' % (species.weight)
            if (i<self.nSpecies-1):
               text += ',  '
            else:
               text += '};  '
            self._write(text + self.line('%s' % species.symbol))
        self._outdent()
        self._write()

        self._write('AMREX_GPU_HOST_DEVICE')
        self._write('void get_imw(double imw_new[]){')
        ##self._write('#pragma unroll')
        self._indent()
        self._write('for(int i = 0; i<%d; ++i) imw_new[i] = imw[i];' %nSpecies )
        self._outdent()
        self._write('}')
        self._write()

        self._write(self.line(' TODO: check necessity because redundant with CKWT'))
        self._write('AMREX_GPU_HOST_DEVICE')
        self._write('void get_mw(double mw_new[]){')
        ##self._write('#pragma unroll')
        self._indent()
        self._write('for(int i = 0; i<%d; ++i) mw_new[i] = molecular_weights[i];' %nSpecies )
        self._outdent()
        self._write('}')
        self._write()

        self._write()

        return


    def _ckinit_QSS(self, mechanism):

        nElement = len(mechanism.element())
        nSpecies = len(mechanism.species())
        nReactions = len(mechanism.reaction())
        
        self._write('#ifndef AMREX_USE_CUDA')
        self._write(self.line(' Initializes parameter database'))
        self._write('void CKINIT'+sym+'()')
        self._write('{')
        self._write()

        self._indent()

        # build reverse reaction map
        rmap = {}
        for i, reaction in zip(range(nReactions), mechanism.reaction()):
            rmap[reaction.orig_id-1] = i

        for j in range(nReactions):
            reaction = mechanism.reaction()[rmap[j]]
            id = reaction.id - 1

            ki = []
            nu = []

            ki_qss = []
            nu_qss = []

            
            for symbol, coefficient in reaction.reactants:
                # print symbol, mechanism.species(symbol), mechanism.species(symbol).id
                if symbol in self.qss_list:
                    ki_qss.append(mechanism.qss_species(symbol).id)
                    nu_qss.append(-coefficient)
                else: 
                    ki.append(mechanism.species(symbol).id)
                    nu.append(-coefficient)
            for symbol, coefficient in reaction.products:
                if symbol in self.qss_list:
                    ki_qss.append(mechanism.qss_species(symbol).id)
                    nu_qss.append(coefficient)
                else:
                    ki.append(mechanism.species(symbol).id)
                    nu.append(coefficient)

            self._write("// (%d):  %s" % (reaction.orig_id - 1, reaction.equation()))
            kistr = "{" + ','.join(str(x) for x in ki) + "}"
            nustr = "{" + ','.join(str(x) for x in nu) + "}"
            kiqss_str = "{" + ','.join(str(x) for x in ki_qss) + "}"
            nuqss_str = "{" + ','.join(str(x) for x in nu_qss) + "}"
            self._write("kiv[%d] = %s;" % (id,kistr))
            self._write("nuv[%d] = %s;" % (id,nustr))
            self._write("kiv_qss[%d] = %s;" % (id,kiqss_str))
            self._write("nuv_qss[%d] = %s;" % (id,nuqss_str))

            A, beta, E = reaction.arrhenius
            self._write("// (%d):  %s" % (reaction.orig_id - 1, reaction.equation()))
            self._write("fwd_A[%d]     = %.17g;" % (id,A))
            self._write("fwd_beta[%d]  = %.17g;" % (id,beta))
            self._write("fwd_Ea[%d]    = %.17g;" % (id,E))

            thirdBody = reaction.thirdBody
            low = reaction.low

            if (reaction.rev):
                Ar, betar, Er = reaction.rev
                self._write("rev_A[%d]     = %.17g;" % (id,Ar))
                self._write("rev_beta[%d]  = %.17g;" % (id,betar))
                self._write("rev_Ea[%d]    = %.17g;" % (id,Er))
                dim_rev       = self._phaseSpaceUnits(reaction.products)
                if not thirdBody:
                    uc_rev = self._prefactorUnits(reaction.units["prefactor"], 1-dim_rev)
                elif not low:
                    uc_rev = self._prefactorUnits(reaction.units["prefactor"], -dim_rev)
                else:
                    uc_rev = self._prefactorUnits(reaction.units["prefactor"], 1-dim_rev)
                self._write("prefactor_units_rev[%d]  = %.17g;" % (id,uc_rev.value))
                aeuc_rev = self._activationEnergyUnits(reaction.units["activation"])
                self._write("activation_units_rev[%d] = %.17g;" % (id,aeuc_rev / Rc / kelvin))

            if (len(reaction.ford) > 0) :
                if (reaction.rev):
                    print '\n\n ***** WARNING: Reac is FORD and REV. Results might be wrong !\n'
                dim = self._phaseSpaceUnits(reaction.ford)
            else:
                dim = self._phaseSpaceUnits(reaction.reactants)

            if not thirdBody:
                uc = self._prefactorUnits(reaction.units["prefactor"], 1-dim) # Case 3 !PD, !TB
            elif not low:
                uc = self._prefactorUnits(reaction.units["prefactor"], -dim) # Case 2 !PD, TB
            else:
                uc = self._prefactorUnits(reaction.units["prefactor"], 1-dim) # Case 1 PD, TB
                low_A, low_beta, low_E = low
                self._write("low_A[%d]     = %.17g;" % (id,low_A))
                self._write("low_beta[%d]  = %.17g;" % (id,low_beta))
                self._write("low_Ea[%d]    = %.17g;" % (id,low_E))
                if reaction.troe:
                    troe = reaction.troe
                    ntroe = len(troe)
                    is_troe = True
                    self._write("troe_a[%d]    = %.17g;" % (id,troe[0]))
                    if ntroe>1:
                        self._write("troe_Tsss[%d] = %.17g;" % (id,troe[1]))
                    if ntroe>2:
                        self._write("troe_Ts[%d]   = %.17g;" % (id,troe[2]))
                    if ntroe>3:
                        self._write("troe_Tss[%d]  = %.17g;" % (id,troe[3]))
                    self._write("troe_len[%d]  = %d;" % (id,ntroe))
                if reaction.sri:
                    sri = reaction.sri
                    nsri = len(sri)
                    is_sri = True
                    self._write("sri_a[%d]     = %.17g;" % (id,sri[0]))
                    if nsri>1:
                        self._write("sri_b[%d]     = %.17g;" % (id,sri[1]))
                    if nsri>2:
                        self._write("sri_c[%d]     = %.17g;" % (id,sri[2]))
                    if nsri>3:
                        self._write("sri_d[%d]     = %.17g;" % (id,sri[3]))
                    if nsri>4:
                        self._write("sri_e[%d]     = %.17g;" % (id,sri[4]))
                    self._write("sri_len[%d]   = %d;" % (id,nsri))

            self._write("prefactor_units[%d]  = %.17g;" % (id,uc.value))
            aeuc = self._activationEnergyUnits(reaction.units["activation"])
            self._write("activation_units[%d] = %.17g;" % (id,aeuc / Rc / kelvin))
            self._write("phase_units[%d]      = pow(10,-%f);" % (id,dim*6))

            if low:
                self._write("is_PD[%d] = 1;" % (id) )
            else:
                self._write("is_PD[%d] = 0;" % (id) )


            if thirdBody:
                efficiencies = reaction.efficiencies
                if (len(efficiencies) > 0):
                    self._write("nTB[%d] = %d;" % (id, len(efficiencies)))
                    self._write("TB[%d] = (double *) malloc(%d * sizeof(double));" % (id, len(efficiencies)))
                    self._write("TBid[%d] = (int *) malloc(%d * sizeof(int));" % (id, len(efficiencies)))
                    for i, eff in enumerate(efficiencies):
                        symbol, efficiency = eff
                        self._write("TBid[%d][%d] = %.17g; TB[%d][%d] = %.17g; // %s"
                                    % (id, i, mechanism.species(symbol).id, id, i, efficiency, symbol ))
                else:
                    self._write("nTB[%d] = 0;" % (id))
            else:
                self._write("nTB[%d] = 0;" % (id))

            self._write()

        self._outdent()
        self._write("}")
        self._write()

        self._write()
        self._write(self.line(' Finalizes parameter database'))
        self._write('void CKFINALIZE()')
        self._write('{')
        self._write('  for (int i=0; i<%d; ++i) {' % (nReactions))
        self._write('    free(TB[i]); TB[i] = 0; ')
        self._write('    free(TBid[i]); TBid[i] = 0;')
        self._write('    nTB[i] = 0;')
        self._write('  }')
        self._write('}')
        self._write()

        self._write('#else')

        self._write(self.line(' TODO: Remove on GPU, right now needed by chemistry_module on FORTRAN'))
        self._write('AMREX_GPU_HOST_DEVICE void CKINIT'+sym+'()')
        self._write('{')
        self._write('}')
        self._write()
        self._write('AMREX_GPU_HOST_DEVICE void CKFINALIZE()')
        self._write('{')
        self._write('}')
        self._write()

        self._write('#endif')

        return


    ################################
    # CHEMKIN WRAPPERS
    ################################

    def _ckindx(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('A few mechanism parameters'))
        self._write('void CKINDX'+sym+'(int * mm, int * kk, int * ii, int * nfit)')
        self._write('{')
        self._indent()
        self._write('*mm = %d;' % len(mechanism.element()))
        self._write('*kk = %d;' % len(mechanism.species()))
        self._write('*ii = %d;' % len(mechanism.reaction()))
        self._write('*nfit = -1; ' + self.line(
            'Why do you need this anyway ? '))
        
        # done
        self._outdent()
        self._write('}')
        return


    def _cksyme_str(self, mechanism):
        nElement = len(mechanism.element())
        self._write()
        self._write()
        self._write(
            self.line(' Returns the vector of strings of element names'))
        self._write('void CKSYME_STR'+sym+'(amrex::Vector<std::string>& ename)')
        self._write('{')
        self._indent()
        self._write('ename.resize(%d);' % nElement)
        for element in mechanism.element():
            self._write('ename[%d] = "%s";' %(element.id, element.symbol))
        # done
        self._outdent()
        self._write('}')
        return


    def _cksyms_str(self, mechanism):
        nSpecies = len(mechanism.species())  
        self._write() 
        self._write()
        self._write(
            self.line(' Returns the vector of strings of species names'))
        self._write('void CKSYMS_STR'+sym+'(amrex::Vector<std::string>& kname)')
        self._write('{')
        self._indent()
        self._write('kname.resize(%d);' % nSpecies)
        for species in mechanism.species():
            self._write('kname[%d] = "%s";' %(species.id, species.symbol))

        self._outdent() 
        self._write('}') 
        return


    def _ckrp(self, mechanism):
        self._write()
        self._write()
        self._write(
            self.line(' Returns R, Rc, Patm' ))
        self._write('void CKRP'+sym+'(double *  ru, double *  ruc, double *  pa)')
        self._write('{')
        self._indent()
        
        self._write(' *ru  = %1.14e; ' % (R * mole * kelvin / erg))
        self._write(' *ruc = %.20f; ' % (Rc * mole * kelvin / cal))
        self._write(' *pa  = %g; ' % (Patm) )
        
        # done
        self._outdent()
        self._write('}')
        return
        

    def _ckpx(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('Compute P = rhoRT/W(x)'))
        self._write('void CKPX'+sym+'(double *  rho, double *  T, double *  x, double *  P)')
        self._write('{')
        self._indent()

        self._write('double XW = 0;'+
                    self.line(' To hold mean molecular wt'))
        
        # molecular weights of all species
        for species in self.species:
            self._write('XW += x[%d]*%f; ' % (
                species.id, species.weight) + self.line('%s' % species.symbol))

        self._write(
            '*P = *rho * %g * (*T) / XW; ' % (R*kelvin*mole/erg)
            + self.line('P = rho*R*T/W'))
        
        self._write()
        self._write('return;')
        self._outdent()
        self._write('}')
        return


    def _ckpy(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('Compute P = rhoRT/W(y)'))
        self._write('AMREX_GPU_HOST_DEVICE void CKPY'+sym+'(double *  rho, double *  T, double *  y,  double *  P)')
        self._write('{')
        self._indent()

        self._write('double YOW = 0;'+self.line(' for computing mean MW'))
        
        # molecular weights of all species
        for species in self.species:
            self._write('YOW += y[%d]*imw[%d]; ' % (
                species.id, species.id) + self.line('%s' % species.symbol))

        self.line('YOW holds the reciprocal of the mean molecular wt')
        self._write(
            '*P = *rho * %g * (*T) * YOW; ' % (R*kelvin*mole/erg)
            + self.line('P = rho*R*T/W'))
        
        
        self._write()
        self._write('return;')
        self._outdent()
        self._write('}')
        return 


    def _ckpc(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('Compute P = rhoRT/W(c)'))
        self._write('void CKPC'+sym+'(double *  rho, double *  T, double *  c,  double *  P)')
        
        self._write('{')
        self._indent()

        self._write('int id; ' + self.line('loop counter'))
        self._write(self.line('See Eq 5 in CK Manual'))
        self._write('double W = 0;')
        self._write('double sumC = 0;')
        
        # molecular weights of all species
        for species in self.species:
            self._write('W += c[%d]*%f; ' % (
                species.id, species.weight) + self.line('%s' % species.symbol))

        self._write()
        nSpecies = len(mechanism.species())
        self._write('for (id = 0; id < %d; ++id) {' % nSpecies)
        self._indent()
        self._write('sumC += c[id];')
        self._outdent()
        self._write('}')

        self.line('W/sumC holds the mean molecular wt')
        self._write(
            '*P = *rho * %g * (*T) * sumC / W; ' % (R*kelvin*mole/erg)
            + self.line('P = rho*R*T/W'))
        
        self._write()
        self._write('return;')
        self._outdent()
        self._write('}')
        return 


    def _ckrhox(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('Compute rho = PW(x)/RT'))
        self._write('void CKRHOX'+sym+'(double *  P, double *  T, double *  x,  double *  rho)')
        self._write('{')
        self._indent()

        self._write('double XW = 0;'+
                    self.line(' To hold mean molecular wt'))
        
        # molecular weights of all species
        for species in self.species:
            self._write('XW += x[%d]*%f; ' % (
                species.id, species.weight) + self.line('%s' % species.symbol))

        self._write(
            '*rho = *P * XW / (%g * (*T)); ' % (R*kelvin*mole/erg)
            + self.line('rho = P*W/(R*T)'))
        
        self._write()
        self._write('return;')
        self._outdent()
        self._write('}')
        return


    def _ckrhoy(self, mechanism):
        species = self.species
        nSpec = len(species)
        self._write()
        self._write()
        self._write(self.line('Compute rho = P*W(y)/RT'))
        self._write('AMREX_GPU_HOST_DEVICE void CKRHOY'+sym+'(double *  P, double *  T, double *  y,  double *  rho)')
        self._write('{')
        self._indent()
        self._write('double YOW = 0;')
        self._write('double tmp[%d];' % (nSpec))
        self._write('')
        self._write('for (int i = 0; i < %d; i++)' % (nSpec))
        self._write('{')
        self._indent()
        self._write('tmp[i] = y[i]*imw[i];')
        self._outdent()
        self._write('}')
        self._write('for (int i = 0; i < %d; i++)' % (nSpec))
        self._write('{')
        self._indent()
        self._write('YOW += tmp[i];')
        self._outdent()
        self._write('}')
        self._write('')
        self._write('*rho = *P / (%g * (*T) * YOW);' % (R * mole * kelvin / erg) + self.line('rho = P*W/(R*T)'))
        self._write('return;')
        self._outdent()
        self._write('}')
        return 


    def _ckrhoc(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('Compute rho = P*W(c)/(R*T)'))
        self._write('void CKRHOC'+sym+'(double *  P, double *  T, double *  c,  double *  rho)')
        
        self._write('{')
        self._indent()

        self._write('int id; ' + self.line('loop counter'))
        self._write(self.line('See Eq 5 in CK Manual'))
        self._write('double W = 0;')
        self._write('double sumC = 0;')
        
        # molecular weights of all species
        for species in self.species:
            self._write('W += c[%d]*%f; ' % (
                species.id, species.weight) + self.line('%s' % species.symbol))

        self._write()
        self._write('for (id = 0; id < %d; ++id) {' % self.nSpecies)
        self._indent()
        self._write('sumC += c[id];')
        self._outdent()
        self._write('}')

        self.line('W/sumC holds the mean molecular wt')
        self._write(
            '*rho = *P * W / (sumC * (*T) * %g); ' % (R*kelvin*mole/erg)
            + self.line('rho = PW/(R*T)'))
        
        self._write()
        self._write('return;')
        self._outdent()
        self._write('}')
        return 


    def _ckwt(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('get molecular weight for all species'))
        self._write('void CKWT'+sym+'( double *  wt)')
        self._write('{')
        self._indent()
        # call molecularWeight
        self._write('get_mw(wt);')
        self._outdent()
        self._write('}')
        return

      
    def _ckawt(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('get atomic weight for all elements'))
        self._write('void CKAWT'+sym+'( double *  awt)')
        self._write('{')
        self._indent()
        # call atomicWeight
        self._write('atomicWeight(awt);')
        self._outdent()
        self._write('}')
        self._write()
        return


    def _ckmmwy(self, mechanism):
        self._write()
        self._write(self.line('given y[species]: mass fractions'))
        self._write(self.line('returns mean molecular weight (gm/mole)'))
        self._write('AMREX_GPU_HOST_DEVICE void CKMMWY'+sym+'(double *  y,  double *  wtm)')
        self._write('{')
        self._indent()
        species = self.species
        nSpec = len(species)
        self._write('double YOW = 0;')
        self._write('double tmp[%d];' % (nSpec))
        self._write('')
        self._write('for (int i = 0; i < %d; i++)' % (nSpec))
        self._write('{')
        self._indent()
        self._write('tmp[i] = y[i]*imw[i];')
        self._outdent()
        self._write('}')
        self._write('for (int i = 0; i < %d; i++)' % (nSpec))
        self._write('{')
        self._indent()
        self._write('YOW += tmp[i];')
        self._outdent()
        self._write('}')
        self._write('')
        self._write('*wtm = 1.0 / YOW;')
        self._write('return;')
        self._outdent()
        self._write('}')

        return 
 

    def _ckmmwx(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('given x[species]: mole fractions'))
        self._write(self.line('returns mean molecular weight (gm/mole)'))
        self._write('void CKMMWX'+sym+'(double *  x,  double *  wtm)')
        self._write('{')
        self._indent()
        self._write('double XW = 0;'+self.line(' see Eq 4 in CK Manual'))
        # molecular weights of all species
        for species in self.species:
            self._write('XW += x[%d]*%f; ' % (
                species.id, species.weight) + self.line('%s' % species.symbol))
        self._write('*wtm = XW;')
        self._write()
        self._write('return;')
        self._outdent()
        self._write('}')
        return 

 
    def _ckmmwc(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('given c[species]: molar concentration'))
        self._write(self.line('returns mean molecular weight (gm/mole)'))
        self._write('void CKMMWC'+sym+'(double *  c,  double *  wtm)')
        self._write('{')
        self._indent()
        self._write('int id; ' + self.line('loop counter'))
        self._write(self.line('See Eq 5 in CK Manual'))
        self._write('double W = 0;')
        self._write('double sumC = 0;')
        # molecular weights of all species
        for species in self.species:
            self._write('W += c[%d]*%f; ' % (
                species.id, species.weight) + self.line('%s' % species.symbol))
        self._write()
        self._write('for (id = 0; id < %d; ++id) {' % self.nSpecies)
        self._indent()
        self._write('sumC += c[id];')
        self._outdent()
        self._write('}')
        self._write(self.line(' CK provides no guard against divison by zero'))
        self._write('*wtm = W/sumC;')
        self._write()
        self._write('return;')
        self._outdent()
        self._write('}')
        return 


    def _ckytx(self, mechanism):
        self._write()
        self._write()
        self._write(self.line(
            'convert y[species] (mass fracs) to x[species] (mole fracs)'))
        self._write('AMREX_GPU_HOST_DEVICE void CKYTX'+sym+'(double *  y,  double *  x)')
        self._write('{')
        self._indent()

        species = self.species
        nSpec = len(species)
        self._write('double YOW = 0;')
        self._write('double tmp[%d];' % (nSpec))
        self._write('')
        self._write('for (int i = 0; i < %d; i++)' % (nSpec))
        self._write('{')
        self._indent()
        self._write('tmp[i] = y[i]*imw[i];')
        self._outdent()
        self._write('}')
        self._write('for (int i = 0; i < %d; i++)' % (nSpec))
        self._write('{')
        self._indent()
        self._write('YOW += tmp[i];')
        self._outdent()
        self._write('}')
        self._write('')
        self._write('double YOWINV = 1.0/YOW;')
        self._write('')
        self._write('for (int i = 0; i < %d; i++)' % (nSpec))
        self._write('{')
        self._indent()
        self._write('x[i] = y[i]*imw[i]*YOWINV;')
        self._outdent()
        self._write('}')
        self._write('return;')
        self._outdent()
        self._write('}')
        return 


    def _ckytcp(self, mechanism):
        self._write()
        self._write()
        self._write(self.line(
            'convert y[species] (mass fracs) to c[species] (molar conc)'))
        self._write('void CKYTCP'+sym+'(double *  P, double *  T, double *  y,  double *  c)')
        self._write('{')
        self._indent()

        species = self.species
        nSpec = len(species)
        self._write('double YOW = 0;')
        self._write('double PWORT;')
        self._write('')
        self._write(self.line('Compute inverse of mean molecular wt first'))
        self._write('for (int i = 0; i < %d; i++)' % (nSpec))
        self._write('{')
        self._indent()
        self._write('c[i] = y[i]*imw[i];')
        self._outdent()
        self._write('}')
        self._write('for (int i = 0; i < %d; i++)' % (nSpec))
        self._write('{')
        self._indent()
        self._write('YOW += c[i];')
        self._outdent()
        self._write('}')
        self._write('')
        self._write(self.line('PW/RT (see Eq. 7)'))
        self._write('PWORT = (*P)/(YOW * %g * (*T)); ' % (R*kelvin*mole/erg) )

        # now compute conversion
        self._write(self.line('Now compute conversion'))
        self._write('')
        self._write('for (int i = 0; i < %d; i++)' % (nSpec))
        self._write('{')
        self._indent()
        self._write('c[i] = PWORT * y[i] * imw[i];')
        self._outdent()
        self._write('}')
        self._write('return;')
        self._outdent()
        self._write('}')
        return 


    def _ckytcr(self, mechanism):
        self._write()
        self._write()
        self._write(self.line(
            'convert y[species] (mass fracs) to c[species] (molar conc)'))
        self._write('AMREX_GPU_HOST_DEVICE void CKYTCR'+sym+'(double *  rho, double *  T, double *  y,  double *  c)')
        self._write('{')
        self._indent()
        species = self.species
        nSpec = len(species)
        self._write('for (int i = 0; i < %d; i++)' % (nSpec))
        self._write('{')
        self._indent()
        self._write('c[i] = (*rho)  * y[i] * imw[i];')
        self._outdent()
        self._write('}')
        self._outdent()
        self._write('}')
        return 


    def _ckxty(self, mechanism):
        self._write()
        self._write()
        self._write(self.line(
            'convert x[species] (mole fracs) to y[species] (mass fracs)'))
        self._write('AMREX_GPU_HOST_DEVICE void CKXTY'+sym+'(double *  x,  double *  y)')
        self._write('{')
        self._indent()

        self._write('double XW = 0; '+self.line('See Eq 4, 9 in CK Manual'))
        
        # compute mean molecular weight first (eq 3)
        self._write(self.line('Compute mean molecular wt first'))
        for species in self.species:
            self._write('XW += x[%d]*%f; ' % (
                species.id, species.weight) + self.line('%s' % species.symbol))
 
        # now compute conversion
        self._write(self.line('Now compute conversion'))
        self._write('double XWinv = 1.0/XW;')
        for species in self.species:
            self._write('y[%d] = x[%d]*%f*XWinv; ' % (
                species.id, species.id, species.weight) )

        self._write()
        self._write('return;')
        self._outdent()
        self._write('}')
        return 

 
    def _ckxtcp(self, mechanism):
        self._write()
        self._write()
        self._write(self.line(
            'convert x[species] (mole fracs) to c[species] (molar conc)'))
        self._write('void CKXTCP'+sym+'(double *  P, double *  T, double *  x,  double *  c)')
        self._write('{')
        self._indent()

        self._write('int id; ' + self.line('loop counter'))
        self._write('double PORT = (*P)/(%g * (*T)); ' % (R*kelvin*mole/erg) +
                    self.line('P/RT'))
        # now compute conversion
        self._write()
        self._write(self.line('Compute conversion, see Eq 10'))
        self._write('for (id = 0; id < %d; ++id) {' % self.nSpecies)
        self._indent()
        self._write('c[id] = x[id]*PORT;')
        self._outdent()
        self._write('}')

        self._write()
        self._write('return;')
        self._outdent()
        self._write('}')
        return 
 

    def _ckxtcr(self, mechanism):
        self._write()
        self._write()
        self._write(self.line(
            'convert x[species] (mole fracs) to c[species] (molar conc)'))
        self._write('void CKXTCR'+sym+'(double *  rho, double *  T, double *  x, double *  c)')
        self._write('{')
        self._indent()

        self._write('int id; ' + self.line('loop counter'))
        self._write('double XW = 0; '+self.line('See Eq 4, 11 in CK Manual'))
        self._write('double ROW; ')
        
        # compute mean molecular weight first (eq 3)
        self._write(self.line('Compute mean molecular wt first'))
        for species in self.species:
            self._write('XW += x[%d]*%f; ' % (
                species.id, species.weight) + self.line('%s' % species.symbol))

        # now compute conversion
        self._write('ROW = (*rho) / XW;')
        self._write()
        self._write(self.line('Compute conversion, see Eq 11'))
        self._write('for (id = 0; id < %d; ++id) {' % self.nSpecies)
        self._indent()
        self._write('c[id] = x[id]*ROW;')
        self._outdent()
        self._write('}')

        self._write()
        self._write('return;')
        self._outdent()
        self._write('}')
        return 

 
    def _ckctx(self, mechanism):
        self._write()
        self._write()
        self._write(self.line(
            'convert c[species] (molar conc) to x[species] (mole fracs)'))
        self._write('void CKCTX'+sym+'(double *  c, double *  x)')
        self._write('{')
        self._indent()

        self._write('int id; ' + self.line('loop counter'))
        self._write('double sumC = 0; ')

        self._write()
        self._write(self.line('compute sum of c '))
        self._write('for (id = 0; id < %d; ++id) {' % self.nSpecies)
        self._indent()
        self._write('sumC += c[id];')
        self._outdent()
        self._write('}')

        # now compute conversion
        self._write()
        self._write(self.line(' See Eq 13 '))
        self._write('double sumCinv = 1.0/sumC;')
        self._write('for (id = 0; id < %d; ++id) {' % self.nSpecies)
        self._indent()
        self._write('x[id] = c[id]*sumCinv;')
        self._outdent()
        self._write('}')

        self._write()
        self._write('return;')
        self._outdent()
        self._write('}')
        return 


    def _ckcty(self, mechanism):
        self._write()
        self._write()
        self._write(self.line(
            'convert c[species] (molar conc) to y[species] (mass fracs)'))
        self._write('void CKCTY'+sym+'(double *  c, double *  y)')
        self._write('{')
        self._indent()

        self._write('double CW = 0; '+self.line('See Eq 12 in CK Manual'))
        
        # compute denominator in eq 12
        self._write(self.line('compute denominator in eq 12 first'))
        for species in self.species:
            self._write('CW += c[%d]*%f; ' % (
                species.id, species.weight) + self.line('%s' % species.symbol))

        # now compute conversion
        self._write(self.line('Now compute conversion'))
        self._write('double CWinv = 1.0/CW;')
        for species in self.species:
            self._write('y[%d] = c[%d]*%f*CWinv; ' % (
                species.id, species.id, species.weight) )

        self._write()
        self._write('return;')
        self._outdent()
        self._write('}')
        return 


    # POLYFITS #
    def _ckcpor(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('get Cp/R as a function of T '))
        self._write(self.line('for all species (Eq 19)'))
        self._write('void CKCPOR'+sym+'(double *  T, double *  cpor)')
        self._write('{')
        self._indent()

        # get temperature cache
        self._write(
            'double tT = *T; '
            + self.line('temporary temperature'))
        self._write(
            'double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; '
            + self.line('temperature cache'))
        
        # call routine
        self._write('cp_R(cpor, tc);')
        
        self._outdent()
        self._write('}')
        return

    
    def _ckhort(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('get H/RT as a function of T '))
        self._write(self.line('for all species (Eq 20)'))
        self._write('void CKHORT'+sym+'(double *  T, double *  hort)')
        self._write('{')
        self._indent()

        # get temperature cache
        self._write(
            'double tT = *T; '
            + self.line('temporary temperature'))
        self._write(
            'double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; '
            + self.line('temperature cache'))
        
        # call routine
        self._write('speciesEnthalpy(hort, tc);')
        
        self._outdent()
        self._write('}')
        return

 
    def _cksor(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('get S/R as a function of T '))
        self._write(self.line('for all species (Eq 21)'))
        self._write('void CKSOR'+sym+'(double *  T, double *  sor)')
        self._write('{')
        self._indent()

        # get temperature cache
        self._write(
            'double tT = *T; '
            + self.line('temporary temperature'))
        self._write(
            'double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; '
            + self.line('temperature cache'))
        
        # call routine
        self._write('speciesEntropy(sor, tc);')
        
        self._outdent()
        self._write('}')
        return

    # POLYFITS #


    def _ckcvms(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('Returns the specific heats at constant volume'))
        self._write(self.line('in mass units (Eq. 29)'))
        self._write('AMREX_GPU_HOST_DEVICE void CKCVMS'+sym+'(double *  T,  double *  cvms)')
        self._write('{')
        self._indent()

        # get temperature cache
        self._write(
            'double tT = *T; '
            + self.line('temporary temperature'))
        self._write(
            'double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; '
            + self.line('temperature cache'))
        
        # call routine
        self._write('cv_R(cvms, tc);')

        # convert cv/R to cv with mass units
        self._write(self.line('multiply by R/molecularweight'))
        for species in self.species:
            ROW = (R*kelvin*mole/erg) / species.weight
            self._write('cvms[%d] *= %20.15e; ' % (
                species.id, ROW) + self.line('%s' % species.symbol))

        self._outdent()

        self._write('}')

        return


    def _ckcpms(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('Returns the specific heats at constant pressure'))
        self._write(self.line('in mass units (Eq. 26)'))
        self._write('AMREX_GPU_HOST_DEVICE void CKCPMS'+sym+'(double *  T,  double *  cpms)')
        self._write('{')
        self._indent()

        # get temperature cache
        self._write(
            'double tT = *T; '
            + self.line('temporary temperature'))
        self._write(
            'double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; '
            + self.line('temperature cache'))
        
        # call routine
        self._write('cp_R(cpms, tc);')
        

        # convert cp/R to cp with mass units
        self._write(self.line('multiply by R/molecularweight'))
        for species in self.species:
            ROW = (R*kelvin*mole/erg) / species.weight
            self._write('cpms[%d] *= %20.15e; ' % (
                species.id, ROW) + self.line('%s' % species.symbol))

        self._outdent()
        self._write('}')
        return


    def _ckums(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('Returns internal energy in mass units (Eq 30.)'))
        self._write('AMREX_GPU_HOST_DEVICE void CKUMS'+sym+'(double *  T,  double *  ums)')
        self._write('{')
        self._indent()

        # get temperature cache
        self._write(
            'double tT = *T; '
            + self.line('temporary temperature'))
        self._write(
            'double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; '
            + self.line('temperature cache'))
        self._write(
            'double RT = %g*tT; ' % (R*kelvin*mole/erg)
            + self.line('R*T'))
        
        # call routine
        self._write('speciesInternalEnergy(ums, tc);')
        
        species = self.species
        nSpec = len(species)
        self._write('for (int i = 0; i < %d; i++)' % (nSpec))
        self._write('{')
        self._indent()
        self._write('ums[i] *= RT*imw[i];')
        self._outdent()
        self._write('}')
        self._outdent()

        self._write('}')

        return


    def _ckhms(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('Returns enthalpy in mass units (Eq 27.)'))
        self._write('AMREX_GPU_HOST_DEVICE void CKHMS'+sym+'(double *  T,  double *  hms)')
        self._write('{')
        self._indent()

        # get temperature cache
        self._write(
            'double tT = *T; '
            + self.line('temporary temperature'))
        self._write(
            'double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; '
            + self.line('temperature cache'))
        self._write(
            'double RT = %g*tT; ' % (R*kelvin*mole/erg)
            + self.line('R*T'))
        
        # call routine
        self._write('speciesEnthalpy(hms, tc);')
        
        species = self.species
        nSpec = len(species)
        self._write('for (int i = 0; i < %d; i++)' % (nSpec))
        self._write('{')
        self._indent()
        self._write('hms[i] *= RT*imw[i];')
        self._outdent()
        self._write('}')
        self._outdent()
        self._write('}')
        return


    def _ckgms(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('Returns gibbs in mass units (Eq 31.)'))
        self._write('void CKGMS'+sym+'(double *  T,  double *  gms)')
        self._write('{')
        self._indent()

        # get temperature cache
        self._write(
            'double tT = *T; '
            + self.line('temporary temperature'))
        self._write(
            'double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; '
            + self.line('temperature cache'))
        self._write(
            'double RT = %g*tT; ' % (R*kelvin*mole/erg)
            + self.line('R*T'))
        
        # call routine
        self._write('gibbs(gms, tc);')
        
        species = self.species
        nSpec = len(species)
        self._write('for (int i = 0; i < %d; i++)' % (nSpec))
        self._write('{')
        self._indent()
        self._write('gms[i] *= RT*imw[i];')
        self._outdent()
        self._write('}')
        self._outdent()
        self._write('}')
        return


    def _ckams(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('Returns helmholtz in mass units (Eq 32.)'))
        self._write('void CKAMS'+sym+'(double *  T,  double *  ams)')
        self._write('{')
        self._indent()

        # get temperature cache
        self._write(
            'double tT = *T; '
            + self.line('temporary temperature'))
        self._write(
            'double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; '
            + self.line('temperature cache'))
        self._write(
            'double RT = %g*tT; ' % (R*kelvin*mole/erg)
            + self.line('R*T'))
        
        # call routine
        self._write('helmholtz(ams, tc);')
        
        species = self.species
        nSpec = len(species)
        self._write('for (int i = 0; i < %d; i++)' % (nSpec))
        self._write('{')
        self._indent()
        self._write('ams[i] *= RT*imw[i];')
        self._outdent()
        self._write('}')

        self._outdent()
        self._write('}')
        return


    def _cksms(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('Returns the entropies in mass units (Eq 28.)'))
        self._write('void CKSMS'+sym+'(double *  T,  double *  sms)')
        self._write('{')
        self._indent()

        # get temperature cache
        self._write(
            'double tT = *T; '
            + self.line('temporary temperature'))
        self._write(
            'double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; '
            + self.line('temperature cache'))
        
        # call routine
        self._write('speciesEntropy(sms, tc);')
        

        # convert s/R to s with mass units
        self._write(self.line('multiply by R/molecularweight'))
        for species in self.species:
            ROW = (R*kelvin*mole/erg) / species.weight
            self._write('sms[%d] *= %20.15e; ' % (
                species.id, ROW) + self.line('%s' % species.symbol))

        self._outdent()
        self._write('}')
        return


    def _ckcpbl(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('Returns the mean specific heat at CP (Eq. 33)'))
        self._write('void CKCPBL'+sym+'(double *  T, double *  x,  double *  cpbl)')
        self._write('{')
        self._indent()

        self._write('int id; ' + self.line('loop counter'))
        self._write('double result = 0; ')
        
        # get temperature cache
        self._write(
            'double tT = *T; '
            + self.line('temporary temperature'))
        self._write(
            'double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; '
            + self.line('temperature cache'))
        self._write(
            'double cpor[%d]; ' % self.nSpecies + self.line(' temporary storage'))
        
        # call routine
        self._write('cp_R(cpor, tc);')
        
        # dot product
        self._write()
        self._write(self.line('perform dot product'))
        self._write('for (id = 0; id < %d; ++id) {' % self.nSpecies)
        self._indent()
        self._write('result += x[id]*cpor[id];')
        self._outdent()
        self._write('}')

        self._write()
        self._write('*cpbl = result * %g;' % (R*kelvin*mole/erg) )
        
        self._outdent()
        self._write('}')
        return

 
    def _ckcpbs(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('Returns the mean specific heat at CP (Eq. 34)'))
        self._write('AMREX_GPU_HOST_DEVICE void CKCPBS'+sym+'(double *  T, double *  y,  double *  cpbs)')
        self._write('{')
        self._indent()

        self._write('double result = 0; ')
        
        # get temperature cache
        self._write(
            'double tT = *T; '
            + self.line('temporary temperature'))
        self._write(
            'double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; '
            + self.line('temperature cache'))
        self._write(
            'double cpor[%d], tresult[%d]; ' % (self.nSpecies,self.nSpecies) + self.line(' temporary storage'))
        
        # call routine
        self._write('cp_R(cpor, tc);')
        

        species = self.species
        nSpec = len(species)
        self._write('for (int i = 0; i < %d; i++)' % (nSpec))
        self._write('{')
        self._indent()
        self._write('tresult[i] = cpor[i]*y[i]*imw[i];')
        self._outdent()
        self._write('')
        self._write('}')
        self._write('for (int i = 0; i < %d; i++)' % (nSpec))
        self._write('{')
        self._indent()
        self._write('result += tresult[i];')
        self._outdent()
        self._write('}')

        self._write()
        self._write('*cpbs = result * %g;' % (R*kelvin*mole/erg) )
        
        self._outdent()
        self._write('}')
        return


    def _ckcvbl(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('Returns the mean specific heat at CV (Eq. 35)'))
        self._write('void CKCVBL'+sym+'(double *  T, double *  x,  double *  cvbl)')
        self._write('{')
        self._indent()

        self._write('int id; ' + self.line('loop counter'))
        self._write('double result = 0; ')
        
        # get temperature cache
        self._write(
            'double tT = *T; '
            + self.line('temporary temperature'))
        self._write(
            'double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; '
            + self.line('temperature cache'))
        self._write(
            'double cvor[%d]; ' % self.nSpecies + self.line(' temporary storage'))
        
        # call routine
        self._write('cv_R(cvor, tc);')
        
        # dot product
        self._write()
        self._write(self.line('perform dot product'))
        self._write('for (id = 0; id < %d; ++id) {' % self.nSpecies)
        self._indent()
        self._write('result += x[id]*cvor[id];')
        self._outdent()
        self._write('}')

        self._write()
        self._write('*cvbl = result * %g;' % (R*kelvin*mole/erg) )
        
        self._outdent()

        self._write('}')

        return


    def _ckcvbs(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('Returns the mean specific heat at CV (Eq. 36)'))
        self._write('AMREX_GPU_HOST_DEVICE void CKCVBS'+sym+'(double *  T, double *  y,  double *  cvbs)')
        self._write('{')
        self._indent()

        self._write('double result = 0; ')
        
        # get temperature cache
        self._write(
            'double tT = *T; '
            + self.line('temporary temperature'))
        self._write(
            'double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; '
            + self.line('temperature cache'))
        self._write(
            'double cvor[%d]; ' % self.nSpecies + self.line(' temporary storage'))
        
        # call routine
        self._write('cv_R(cvor, tc);')
        
        # do dot product
        self._write(self.line('multiply by y/molecularweight'))
        for species in self.species:
            self._write('result += cvor[%d]*y[%d]*imw[%d]; ' % (
                species.id, species.id, species.id) + self.line('%s' % species.symbol))

        self._write()
        self._write('*cvbs = result * %g;' % (R*kelvin*mole/erg) )
        
        self._outdent()
        self._write('}')
        return


    def _ckhbml(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('Returns the mean enthalpy of the mixture in molar units'))
        self._write('void CKHBML'+sym+'(double *  T, double *  x,  double *  hbml)')
        self._write('{')
        self._indent()

        self._write('int id; ' + self.line('loop counter'))
        self._write('double result = 0; ')
        
        # get temperature cache
        self._write(
            'double tT = *T; '
            + self.line('temporary temperature'))
        self._write(
            'double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; '
            + self.line('temperature cache'))
        self._write(
            'double hml[%d]; ' % self.nSpecies + self.line(' temporary storage'))
        self._write(
            'double RT = %g*tT; ' % (R*kelvin*mole/erg)
            + self.line('R*T'))
        
        # call routine
        self._write('speciesEnthalpy(hml, tc);')
        
        # dot product
        self._write()
        self._write(self.line('perform dot product'))
        self._write('for (id = 0; id < %d; ++id) {' % self.nSpecies)
        self._indent()
        self._write('result += x[id]*hml[id];')
        self._outdent()
        self._write('}')

        self._write()
        self._write('*hbml = result * RT;')
        
        self._outdent()

        self._write('}')

        return
 
 
    def _ckhbms(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('Returns mean enthalpy of mixture in mass units'))
        self._write('AMREX_GPU_HOST_DEVICE void CKHBMS'+sym+'(double *  T, double *  y,  double *  hbms)')
        self._write('{')
        self._indent()

        self._write('double result = 0;')
        
        # get temperature cache
        self._write(
            'double tT = *T; '
            + self.line('temporary temperature'))
        self._write(
            'double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; '
            + self.line('temperature cache'))
        self._write(
            'double hml[%d], tmp[%d]; ' % (self.nSpecies,self.nSpecies) + self.line(' temporary storage'))
        
        self._write(
            'double RT = %g*tT; ' % (R*kelvin*mole/erg)
            + self.line('R*T'))
        
        # call routine
        self._write('speciesEnthalpy(hml, tc);')

        self._write('int id;')
        self._write('for (id = 0; id < %d; ++id) {' % self.nSpecies)
        self._indent()
        self._write('tmp[id] = y[id]*hml[id]*imw[id];')
        self._outdent()
        self._write('}')
        self._write('for (id = 0; id < %d; ++id) {' % self.nSpecies)
        self._indent()
        self._write('result += tmp[id];')
        self._outdent()
        self._write('}')

        self._write()
        # finally, multiply by RT
        self._write('*hbms = result * RT;')
        
        self._outdent()
        self._write('}')
        return


    def _ckubml(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('get mean internal energy in molar units'))
        self._write('void CKUBML'+sym+'(double *  T, double *  x,  double *  ubml)')
        self._write('{')
        self._indent()

        self._write('int id; ' + self.line('loop counter'))
        self._write('double result = 0; ')
        
        # get temperature cache
        self._write(
            'double tT = *T; '
            + self.line('temporary temperature'))
        self._write(
            'double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; '
            + self.line('temperature cache'))
        self._write(
            'double uml[%d]; ' % self.nSpecies + self.line(' temporary energy array'))
        self._write(
            'double RT = %g*tT; ' % (R*kelvin*mole/erg)
            + self.line('R*T'))
        
        # call routine
        self._write('speciesInternalEnergy(uml, tc);')
        
        # dot product
        self._write()
        self._write(self.line('perform dot product'))
        self._write('for (id = 0; id < %d; ++id) {' % self.nSpecies)
        self._indent()
        self._write('result += x[id]*uml[id];')
        self._outdent()
        self._write('}')

        self._write()
        self._write('*ubml = result * RT;')
        
        self._outdent()
        self._write('}')
        return
 

    def _ckubms(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('get mean internal energy in mass units'))
        self._write('AMREX_GPU_HOST_DEVICE void CKUBMS'+sym+'(double *  T, double *  y,  double *  ubms)')
        self._write('{')
        self._indent()

        self._write('double result = 0;')
        
        # get temperature cache
        self._write(
            'double tT = *T; '
            + self.line('temporary temperature'))
        self._write(
            'double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; '
            + self.line('temperature cache'))
        self._write(
            'double ums[%d]; ' % self.nSpecies + self.line(' temporary energy array'))
        
        self._write(
            'double RT = %g*tT; ' % (R*kelvin*mole/erg)
            + self.line('R*T'))
        
        # call routine
        self._write('speciesInternalEnergy(ums, tc);')

        # convert e/RT to e with mass units
        self._write(self.line('perform dot product + scaling by wt'))
        for species in self.species:
            self._write('result += y[%d]*ums[%d]*imw[%d]; ' % (
                species.id, species.id, species.id)
                        + self.line('%s' % species.symbol))

        
        self._write()
        # finally, multiply by RT
        self._write('*ubms = result * RT;')
        
        self._outdent()
        self._write('}')
        return


    def _cksbml(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('get mixture entropy in molar units'))
        self._write('void CKSBML'+sym+'(double *  P, double *  T, double *  x,  double *  sbml)')
        self._write('{')
        self._indent()

        self._write('int id; ' + self.line('loop counter'))
        self._write('double result = 0; ')
        
        # get temperature cache
        self._write(self.line('Log of normalized pressure in cgs units dynes/cm^2 by Patm'))
        self._write( 'double logPratio = log ( *P / 1013250.0 ); ')
        self._write(
            'double tT = *T; '
            + self.line('temporary temperature'))
        self._write(
            'double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; '
            + self.line('temperature cache'))
        self._write(
            'double sor[%d]; ' % self.nSpecies + self.line(' temporary storage'))
        
        
        # call routine
        self._write('speciesEntropy(sor, tc);')
        
        # Equation 42
        self._write()
        self._write(self.line('Compute Eq 42'))
        self._write('for (id = 0; id < %d; ++id) {' % self.nSpecies)
        self._indent()
        self._write('result += x[id]*(sor[id]-log((x[id]+%g))-logPratio);' %
                    smallnum )
        self._outdent()
        self._write('}')

        self._write()
        
        self._write('*sbml = result * %g;' % (R*kelvin*mole/erg) )
        
        self._outdent()
        self._write('}')
        return


    def _cksbms(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('get mixture entropy in mass units'))
        self._write('void CKSBMS'+sym+'(double *  P, double *  T, double *  y,  double *  sbms)')
        self._write('{')
        self._indent()

        self._write('double result = 0; ')
        
        # get temperature cache
        self._write(self.line('Log of normalized pressure in cgs units dynes/cm^2 by Patm'))
        self._write( 'double logPratio = log ( *P / 1013250.0 ); ')
        self._write(
            'double tT = *T; '
            + self.line('temporary temperature'))
        self._write(
            'double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; '
            + self.line('temperature cache'))
        self._write(
            'double sor[%d]; ' % self.nSpecies + self.line(' temporary storage'))
        self._write(
            'double x[%d]; ' % self.nSpecies + self.line(' need a ytx conversion'))

        self._write('double YOW = 0; '+self.line('See Eq 4, 6 in CK Manual'))
        
        
        # compute inverse of mean molecular weight first (eq 3)
        self._write(self.line('Compute inverse of mean molecular wt first'))
        for species in self.species:
            self._write('YOW += y[%d]*imw[%d]; ' % (
                species.id, species.id) + self.line('%s' % species.symbol))
 
        # now to ytx
        self._write(self.line('Now compute y to x conversion'))
        for species in self.species:
            self._write('x[%d] = y[%d]/(%f*YOW); ' % (
                species.id, species.id, species.weight) )
            
        # call routine
        self._write('speciesEntropy(sor, tc);')
        
        # Equation 42 and 43
        self._write(self.line('Perform computation in Eq 42 and 43'))
        for species in self.species:
            self._write('result += x[%d]*(sor[%d]-log((x[%d]+%g))-logPratio);' %
                        (species.id, species.id, species.id, smallnum) )

        self._write(self.line('Scale by R/W'))
        self._write('*sbms = result * %g * YOW;' % (R*kelvin*mole/erg) )
        
        self._outdent()
        self._write('}')
        return


    def _ckgbml(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('Returns mean gibbs free energy in molar units'))
        self._write('void CKGBML'+sym+'(double *  P, double *  T, double *  x,  double *  gbml)')
        self._write('{')
        self._indent()

        self._write('int id; ' + self.line('loop counter'))
        self._write('double result = 0; ')
        
        # get temperature cache
        self._write(self.line('Log of normalized pressure in cgs units dynes/cm^2 by Patm'))
        self._write( 'double logPratio = log ( *P / 1013250.0 ); ')
        self._write(
            'double tT = *T; '
            + self.line('temporary temperature'))
        self._write(
            'double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; '
            + self.line('temperature cache'))
        self._write(
            'double RT = %g*tT; ' % (R*kelvin*mole/erg)
            + self.line('R*T'))
        self._write(
            'double gort[%d]; ' % self.nSpecies + self.line(' temporary storage'))
        
        # call routine
        self._write(self.line('Compute g/RT'))
        self._write('gibbs(gort, tc);')
        
        # Equation 44
        self._write()
        self._write(self.line('Compute Eq 44'))
        self._write('for (id = 0; id < %d; ++id) {' % self.nSpecies)
        self._indent()
        self._write('result += x[id]*(gort[id]+log((x[id]+%g))+logPratio);' %
                    smallnum )
        self._outdent()
        self._write('}')

        self._write()
        
        self._write('*gbml = result * RT;')
        
        self._outdent()
        self._write('}')
        return


    def _ckgbms(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('Returns mixture gibbs free energy in mass units'))
        self._write('void CKGBMS'+sym+'(double *  P, double *  T, double *  y,  double *  gbms)')
        self._write('{')
        self._indent()

        self._write('double result = 0; ')
        
        # get temperature cache
        self._write(self.line('Log of normalized pressure in cgs units dynes/cm^2 by Patm'))
        self._write( 'double logPratio = log ( *P / 1013250.0 ); ')
        self._write(
            'double tT = *T; '
            + self.line('temporary temperature'))
        self._write(
            'double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; '
            + self.line('temperature cache'))
        self._write(
            'double RT = %g*tT; ' % (R*kelvin*mole/erg)
            + self.line('R*T'))
        self._write(
            'double gort[%d]; ' % self.nSpecies + self.line(' temporary storage'))
        self._write(
            'double x[%d]; ' % self.nSpecies + self.line(' need a ytx conversion'))

        self._write(
            'double YOW = 0; '
            + self.line('To hold 1/molecularweight'))
        
        
        # compute inverse of mean molecular weight first (eq 3)
        self._write(self.line('Compute inverse of mean molecular wt first'))
        for species in self.species:
            self._write('YOW += y[%d]*imw[%d]; ' % (
                species.id, species.id) + self.line('%s' % species.symbol))
 
        # now to ytx
        self._write(self.line('Now compute y to x conversion'))
        for species in self.species:
            self._write('x[%d] = y[%d]/(%f*YOW); ' % (
                species.id, species.id, species.weight) )
            
        # call routine
        self._write('gibbs(gort, tc);')
        
        # Equation 42 and 43
        self._write(self.line('Perform computation in Eq 44'))
        for species in self.species:
            self._write('result += x[%d]*(gort[%d]+log((x[%d]+%g))+logPratio);' %
                        (species.id, species.id, species.id, smallnum) )

        self._write(self.line('Scale by RT/W'))
        self._write('*gbms = result * RT * YOW;')
        
        self._outdent()
        self._write('}')
        return
    

    def _ckabml(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('Returns mean helmholtz free energy in molar units'))
        self._write('void CKABML'+sym+'(double *  P, double *  T, double *  x,  double *  abml)')
        self._write('{')
        self._indent()

        self._write('int id; ' + self.line('loop counter'))
        self._write('double result = 0; ')
        
        # get temperature cache
        self._write(self.line('Log of normalized pressure in cgs units dynes/cm^2 by Patm'))
        self._write( 'double logPratio = log ( *P / 1013250.0 ); ')
        self._write(
            'double tT = *T; '
            + self.line('temporary temperature'))
        self._write(
            'double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; '
            + self.line('temperature cache'))
        self._write(
            'double RT = %g*tT; ' % (R*kelvin*mole/erg)
            + self.line('R*T'))
        self._write(
            'double aort[%d]; ' % self.nSpecies + self.line(' temporary storage'))
        
        # call routine
        self._write(self.line('Compute g/RT'))
        self._write('helmholtz(aort, tc);')
        
        # Equation 44
        self._write()
        self._write(self.line('Compute Eq 44'))
        self._write('for (id = 0; id < %d; ++id) {' % self.nSpecies)
        self._indent()
        self._write('result += x[id]*(aort[id]+log((x[id]+%g))+logPratio);' %
                    smallnum )
        self._outdent()
        self._write('}')

        self._write()
        
        self._write('*abml = result * RT;')
        
        self._outdent()
        self._write('}')
        return
    

    def _ckabms(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('Returns mixture helmholtz free energy in mass units'))
        self._write('void CKABMS'+sym+'(double *  P, double *  T, double *  y,  double *  abms)')
        self._write('{')
        self._indent()

        self._write('double result = 0; ')
        
        # get temperature cache
        self._write(self.line('Log of normalized pressure in cgs units dynes/cm^2 by Patm'))
        self._write( 'double logPratio = log ( *P / 1013250.0 ); ')
        self._write(
            'double tT = *T; '
            + self.line('temporary temperature'))
        self._write(
            'double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; '
            + self.line('temperature cache'))
        self._write(
            'double RT = %g*tT; ' % (R*kelvin*mole/erg)
            + self.line('R*T'))
        self._write(
            'double aort[%d]; ' % self.nSpecies + self.line(' temporary storage'))
        self._write(
            'double x[%d]; ' % self.nSpecies + self.line(' need a ytx conversion'))

        self._write(
            'double YOW = 0; '
            + self.line('To hold 1/molecularweight'))
        
        
        # compute inverse of mean molecular weight first (eq 3)
        self._write(self.line('Compute inverse of mean molecular wt first'))
        for species in self.species:
            self._write('YOW += y[%d]*imw[%d]; ' % (
                species.id, species.id) + self.line('%s' % species.symbol))
 
        # now to ytx
        self._write(self.line('Now compute y to x conversion'))
        for species in self.species:
            self._write('x[%d] = y[%d]/(%f*YOW); ' % (
                species.id, species.id, species.weight) )
            
        # call routine
        self._write('helmholtz(aort, tc);')
        
        # Equation 42 and 43
        self._write(self.line('Perform computation in Eq 44'))
        for species in self.species:
            self._write('result += x[%d]*(aort[%d]+log((x[%d]+%g))+logPratio);' %
                        (species.id, species.id, species.id, smallnum) )

        self._write(self.line('Scale by RT/W'))
        self._write('*abms = result * RT * YOW;')
        
        self._outdent()
        self._write('}')
        return
    

    def _ckwc(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('compute the production rate for each species'))
        self._write('AMREX_GPU_HOST_DEVICE void CKWC'+sym+'(double *  T, double *  C,  double *  wdot)')
        self._write('{')
        self._indent()

        self._write('int id; ' + self.line('loop counter'))

        # convert C to SI units
        self._write()
        self._write(self.line('convert to SI'))
        self._write('for (id = 0; id < %d; ++id) {' % self.nSpecies)
        self._indent()
        self._write('C[id] *= 1.0e6;')
        self._outdent()
        self._write('}')
        
        # call productionRate
        self._write()
        self._write(self.line('convert to chemkin units'))
        self._write('productionRate(wdot, C, *T);')

        # convert C and wdot to chemkin units
        self._write()
        self._write(self.line('convert to chemkin units'))
        self._write('for (id = 0; id < %d; ++id) {' % self.nSpecies)
        self._indent()
        self._write('C[id] *= 1.0e-6;')
        self._write('wdot[id] *= 1.0e-6;')
        self._outdent()
        self._write('}')
        
        self._outdent()
        self._write('}')
        return


    def _ckwyp(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('Returns the molar production rate of species'))
        self._write(self.line('Given P, T, and mass fractions'))
        self._write('void CKWYP'+sym+'(double *  P, double *  T, double *  y,  double *  wdot)')
        self._write('{')
        self._indent()

        self._write('int id; ' + self.line('loop counter'))

        self._write('double c[%d]; ' % self.nSpecies + self.line('temporary storage'))
        self._write('double YOW = 0; ')
        self._write('double PWORT; ')
        
        # compute inverse of mean molecular weight first (eq 3)
        self._write(self.line('Compute inverse of mean molecular wt first'))
        for species in self.species:
            self._write('YOW += y[%d]*imw[%d]; ' % (
                species.id, species.id) + self.line('%s' % species.symbol))
 
        self._write(self.line('PW/RT (see Eq. 7)'))
        self._write('PWORT = (*P)/(YOW * %g * (*T)); ' % (R*kelvin*mole/erg) )
        
        self._write(self.line('multiply by 1e6 so c goes to SI'))
        self._write('PWORT *= 1e6; ')

        # now compute conversion
        self._write(self.line('Now compute conversion (and go to SI)'))
        for species in self.species:
            self._write('c[%d] = PWORT * y[%d]*imw[%d]; ' % (
                species.id, species.id, species.id) )

        # call productionRate
        self._write()
        self._write(self.line('convert to chemkin units'))
        self._write('productionRate(wdot, c, *T);')

        # convert wdot to chemkin units
        self._write()
        self._write(self.line('convert to chemkin units'))
        self._write('for (id = 0; id < %d; ++id) {' % self.nSpecies)
        self._indent()
        self._write('wdot[id] *= 1.0e-6;')
        self._outdent()
        self._write('}')
        
        self._outdent()
        self._write('}')
        return


    def _ckwxp(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('Returns the molar production rate of species'))
        self._write(self.line('Given P, T, and mole fractions'))
        self._write('void CKWXP'+sym+'(double *  P, double *  T, double *  x,  double *  wdot)')
        self._write('{')
        self._indent()

        self._write('int id; ' + self.line('loop counter'))

        self._write('double c[%d]; ' % self.nSpecies + self.line('temporary storage'))
        
        self._write('double PORT = 1e6 * (*P)/(%g * (*T)); ' % (R*kelvin*mole/erg) +
                    self.line('1e6 * P/RT so c goes to SI units'))
        
        # now compute conversion
        self._write()
        self._write(self.line('Compute conversion, see Eq 10'))
        self._write('for (id = 0; id < %d; ++id) {' % self.nSpecies)
        self._indent()
        self._write('c[id] = x[id]*PORT;')
        self._outdent()
        self._write('}')
        
        # call productionRate
        self._write()
        self._write(self.line('convert to chemkin units'))
        self._write('productionRate(wdot, c, *T);')

        # convert wdot to chemkin units
        self._write()
        self._write(self.line('convert to chemkin units'))
        self._write('for (id = 0; id < %d; ++id) {' % self.nSpecies)
        self._indent()
        self._write('wdot[id] *= 1.0e-6;')
        self._outdent()
        self._write('}')
        
        self._outdent()
        self._write('}')
        return


    def _ckwyr(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('Returns the molar production rate of species'))
        self._write(self.line('Given rho, T, and mass fractions'))
        self._write('AMREX_GPU_HOST_DEVICE void CKWYR'+sym+'(double *  rho, double *  T, double *  y,  double *  wdot)')
        self._write('{')
        self._indent()

        self._write('int id; ' + self.line('loop counter'))

        self._write('double c[%d]; ' % self.nSpecies + self.line('temporary storage'))

        # now compute conversion
        self._write(self.line('See Eq 8 with an extra 1e6 so c goes to SI'))
        for species in self.species:
            self._write('c[%d] = 1e6 * (*rho) * y[%d]*imw[%d]; ' % (
                species.id, species.id, species.id) )
            
        # call productionRate
        self._write()
        self._write(self.line('call productionRate'))
        self._write('productionRate(wdot, c, *T);')

        # convert wdot to chemkin units
        self._write()
        self._write(self.line('convert to chemkin units'))
        self._write('for (id = 0; id < %d; ++id) {' % self.nSpecies)
        self._indent()
        self._write('wdot[id] *= 1.0e-6;')
        self._outdent()
        self._write('}')
        
        self._outdent()
        self._write('}')
        return


    def _ckwxr(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('Returns the molar production rate of species'))
        self._write(self.line('Given rho, T, and mole fractions'))
        self._write('void CKWXR'+sym+'(double *  rho, double *  T, double *  x,  double *  wdot)')
        self._write('{')
        self._indent()

        self._write('int id; ' + self.line('loop counter'))

        self._write('double c[%d]; ' % self.nSpecies + self.line('temporary storage'))
        
        self._write('double XW = 0; '+self.line('See Eq 4, 11 in CK Manual'))
        self._write('double ROW; ')
        
        # compute mean molecular weight first (eq 3)
        self._write(self.line('Compute mean molecular wt first'))
        for species in self.species:
            self._write('XW += x[%d]*%f; ' % (
                species.id, species.weight) + self.line('%s' % species.symbol))

        # now compute conversion
        self._write(self.line('Extra 1e6 factor to take c to SI'))
        self._write('ROW = 1e6*(*rho) / XW;')
        self._write()
        self._write(self.line('Compute conversion, see Eq 11'))
        self._write('for (id = 0; id < %d; ++id) {' % self.nSpecies)
        self._indent()
        self._write('c[id] = x[id]*ROW;')
        self._outdent()
        self._write('}')
        
        # call productionRate
        self._write()
        self._write(self.line('convert to chemkin units'))
        self._write('productionRate(wdot, c, *T);')

        # convert wdot to chemkin units
        self._write()
        self._write(self.line('convert to chemkin units'))
        self._write('for (id = 0; id < %d; ++id) {' % self.nSpecies)
        self._indent()
        self._write('wdot[id] *= 1.0e-6;')
        self._outdent()
        self._write('}')
        
        self._outdent()
        self._write('}')
        return


    def _ckncf(self, mechanism):

        nSpecies  = len(mechanism.species())
        nElement  = len(mechanism.element())

        self._write()
        self._write()
        self._write(self.line('Returns the elemental composition '))
        self._write(self.line('of the speciesi (mdim is num of elements)'))
        self._write('void CKNCF'+sym+'(int * ncf)')
        self._write('{')
        self._indent()

 
        self._write('int id; ' + self.line('loop counter'))
        self._write('int kd = %d; ' % (nElement))
        self._write(self.line('Zero ncf'))
        self._write('for (id = 0; id < kd * %d; ++ id) {' % (self.nSpecies) )
        self._indent()
        self._write(' ncf[id] = 0; ')
        self._outdent()
        self._write('}')
        
        self._write()
        for species in mechanism.species():
           self._write(self.line('%s' % species.symbol))
           for elem, coef in species.composition:
               self._write('ncf[ %d * kd + %d ] = %d; ' % (
                   species.id, mechanism.element(elem).id, coef) +
                       self.line('%s' % elem) )
                           
           self._write()
                            
        # done
        self._outdent()
        self._write('}')
        return

    ################################
    # END CHEMKIN WRAPPERS
    ################################


    ################################
    # MAIN PROD RATE GENERATORS
    ################################

    def _productionRate(self, mechanism):

        nSpecies = len(mechanism.species())
        nReactions = len(mechanism.reaction())

        itroe      = self.reactionIndex[0:2]
        isri       = self.reactionIndex[1:3]
        ilindemann = self.reactionIndex[2:4]
        i3body     = self.reactionIndex[3:5] 
        isimple    = self.reactionIndex[4:6]
        ispecial   = self.reactionIndex[5:7]

        if len(self.reactionIndex) != 7:
            print '\n\nCheck this!!!\n'
            sys.exit(1)
        
        ntroe      = itroe[1]      - itroe[0]
        nsri       = isri[1]       - isri[0]
        nlindemann = ilindemann[1] - ilindemann[0]
        n3body     = i3body[1]     - i3body[0]
        nsimple    = isimple[1]    - isimple[0]
        nspecial   = ispecial[1]   - ispecial[0]

        # OMP stuff
        self._write()
        self._write("#ifndef AMREX_USE_CUDA")
        self._write('static double T_save = -1;')
        self._write('#ifdef _OPENMP')
        self._write('#pragma omp threadprivate(T_save)')
        self._write('#endif')
        self._write()
        self._write('static double k_f_save[%d];' % nReactions)
        self._write('#ifdef _OPENMP')
        self._write('#pragma omp threadprivate(k_f_save)')
        self._write('#endif')
        self._write()
        self._write('static double Kc_save[%d];' % nReactions)
        self._write('#ifdef _OPENMP')
        self._write('#pragma omp threadprivate(Kc_save)')
        self._write('#endif')
        self._write()

        # main function
        self._write()
        self._write(self.line('compute the production rate for each species pointwise on CPU'))
        self._write('void productionRate(double *  wdot, double *  sc, double T)')
        self._write('{')
        self._indent()

        self._write('double tc[] = { log(T), T, T*T, T*T*T, T*T*T*T }; /*temperature cache */')
        self._write('double invT = 1.0 / tc[1];')
        
        self._write()
        self._write('if (T != T_save)')
        self._write('{')
        self._indent()
        self._write('T_save = T;')
        self._write('comp_k_f(tc,invT,k_f_save);');
        self._write('comp_Kc(tc,invT,Kc_save);');
        self._outdent()
        self._write("}")

        self._write()
        self._write('double qdot, q_f[%d], q_r[%d];' % (nReactions,nReactions))
        self._write('comp_qfqr(q_f, q_r, sc, tc, invT);');

        self._write()
        self._write('for (int i = 0; i < %d; ++i) {' % nSpecies)
        self._indent()
        self._write('wdot[i] = 0.0;')
        self._outdent()
        self._write('}')

        for i in range(nReactions):
            self._write()
            self._write("qdot = q_f[%d]-q_r[%d];" % (i,i))
            reaction = mechanism.reaction(id=i)
            all_agents = list(set(reaction.reactants + reaction.products))
            agents = []
            # remove QSS species from agents
            for symbol, coefficient in all_agents:
                if symbol not in self.qss_list:
                    agents.append((symbol, coefficient))
            agents = sorted(agents, key=lambda x: mechanism.species(x[0]).id)
            # note that a species might appear as both reactant and product
            # a species might alos appear twice or more on on each side
            # agents is a set that contains unique (symbol, coefficient)
            for a in agents:
                symbol, coefficient = a
                for b in reaction.reactants:
                    if b == a:
                        if coefficient == 1:
                            self._write("wdot[%d] -= qdot;" % (mechanism.species(symbol).id))
                        else:
                            self._write("wdot[%d] -= %f * qdot;" % (mechanism.species(symbol).id, coefficient))
                for b in reaction.products: 
                    if b == a:
                        if coefficient == 1:
                            self._write("wdot[%d] += qdot;" % (mechanism.species(symbol).id))
                        else:
                            self._write("wdot[%d] += %f * qdot;" % (mechanism.species(symbol).id, coefficient))


        self._write()
        self._write('return;')
        self._outdent()

        self._write('}')

        # k_f function
        self._write()
        self._write('void comp_k_f(double *  tc, double invT, double *  k_f)')
        self._write('{')
        self._indent()
        self._outdent()
        self._write('#ifdef __INTEL_COMPILER')
        self._indent()
        self._write('#pragma simd')
        self._outdent()
        self._write('#endif')
        self._indent()
        self._write('for (int i=0; i<%d; ++i) {' % (nReactions))
        self._indent()
        self._write("k_f[i] = prefactor_units[i] * fwd_A[i]")
        self._write("            * exp(fwd_beta[i] * tc[0] - activation_units[i] * fwd_Ea[i] * invT);")
        self._outdent()
        self._write("};")
        self._write('return;')
        self._outdent()
        self._write('}')

        # Kc
        self._write()
        self._write('void comp_Kc(double *  tc, double invT, double *  Kc)')
        self._write('{')
        self._indent()

        self._write(self.line('compute the Gibbs free energy'))
        self._write('double g_RT[%d], g_RT_qss[%d];' % (nSpecies,self.nQSSspecies))
        self._write('gibbs(g_RT, tc);')
        self._write('gibbs_qss(g_RT_qss, tc);')

        self._write()

        for reaction in mechanism.reaction():
            KcExpArg = self._sortedKcExpArg(mechanism, reaction)
            self._write("Kc[%d] = %s;" % (reaction.id-1,KcExpArg))

        self._write()
        
        self._outdent()
        self._write('#ifdef __INTEL_COMPILER')
        self._indent()
        self._write(' #pragma simd')
        self._outdent()
        self._write('#endif')
        self._indent()
        self._write('for (int i=0; i<%d; ++i) {' % (nReactions))
        self._indent()
        self._write("Kc[i] = exp(Kc[i]);")
        self._outdent()
        self._write("};")

        self._write()

        self._write(self.line('reference concentration: P_atm / (RT) in inverse mol/m^3'))
        self._write('double refC = %g / %g * invT;' % (atm.value, R.value))
        self._write('double refCinv = 1 / refC;')

        self._write()

        for reaction in mechanism.reaction():
            KcConv = self._KcConv(mechanism, reaction)
            if KcConv:
                self._write("Kc[%d] *= %s;" % (reaction.id-1,KcConv))        
        
        self._write()

        self._write('return;')
        self._outdent()
        self._write('}')

        # qdot
        self._write()
        self._write('void comp_qfqr(double *  qf, double *  qr, double *  sc, double * qss_sc, double *  tc, double invT)')
        self._write('{')
        self._indent()

        nclassd = nReactions - nspecial
        nCorr   = n3body + ntroe + nsri + nlindemann

        for i in range(nclassd):
            self._write()
            reaction = mechanism.reaction(id=i)
            self._write(self.line('reaction %d: %s' % (reaction.id, reaction.equation())))
            if (len(reaction.ford) > 0):
                self._write("qf[%d] = %s;" % (i, self._QSSsortedPhaseSpace(mechanism, reaction.ford)))
            else:
                self._write("qf[%d] = %s;" % (i, self._QSSsortedPhaseSpace(mechanism, reaction.reactants)))
            if reaction.reversible:
                self._write("qr[%d] = %s;" % (i, self._QSSsortedPhaseSpace(mechanism, reaction.products)))
            else:
                self._write("qr[%d] = 0.0;" % (i))

        self._write()
        self._write('double T = tc[1];')
        self._write()
        self._write(self.line('compute the mixture concentration'))
        self._write('double mixture = 0.0;')
        self._write('for (int i = 0; i < %d; ++i) {' % nSpecies)
        self._indent()
        self._write('mixture += sc[i];')
        self._outdent()
        self._write('}')

        self._write()
        self._write("double Corr[%d];" % nclassd)
        self._write('for (int i = 0; i < %d; ++i) {' % nclassd)
        self._indent()
        self._write('Corr[i] = 1.0;')
        self._outdent()
        self._write('}')

        if ntroe > 0:
            self._write()
            self._write(self.line(" troe"))
            self._write("{")
            self._indent()
            self._write("double alpha[%d];" % ntroe)
            alpha_d = {}
            for i in range(itroe[0],itroe[1]):
                ii = i - itroe[0]
                reaction = mechanism.reaction(id=i)
                if reaction.thirdBody:
                    alpha = self._enhancement(mechanism, reaction)
                    if alpha in alpha_d:
                        self._write("alpha[%d] = %s;" %(ii,alpha_d[alpha]))
                    else:
                        self._write("alpha[%d] = %s;" %(ii,alpha))
                        alpha_d[alpha] = "alpha[%d]" % ii

            if ntroe >= 4:
                self._outdent()
                self._outdent()
                self._write('#ifdef __INTEL_COMPILER')
                self._indent()
                self._indent()
                self._write(' #pragma simd')
                self._outdent()
                self._outdent()
                self._write('#endif')
                self._indent()
                self._indent()
            self._write("for (int i=%d; i<%d; i++)" %(itroe[0],itroe[1]))
            self._write("{")
            self._indent()
            self._write("double redP, F, logPred, logFcent, troe_c, troe_n, troe, F_troe;")
            self._write("redP = alpha[i-%d] / k_f_save[i] * phase_units[i] * low_A[i] * exp(low_beta[i] * tc[0] - activation_units[i] * low_Ea[i] *invT);" % itroe[0])
            self._write("F = redP / (1.0 + redP);")
            self._write("logPred = log10(redP);")
            self._write('logFcent = log10(')
            self._write('    (fabs(troe_Tsss[i]) > 1.e-100 ? (1.-troe_a[i])*exp(-T/troe_Tsss[i]) : 0.) ')
            self._write('    + (fabs(troe_Ts[i]) > 1.e-100 ? troe_a[i] * exp(-T/troe_Ts[i]) : 0.) ')
            self._write('    + (troe_len[i] == 4 ? exp(-troe_Tss[i] * invT) : 0.) );')
            self._write("troe_c = -.4 - .67 * logFcent;")
            self._write("troe_n = .75 - 1.27 * logFcent;")
            self._write("troe = (troe_c + logPred) / (troe_n - .14*(troe_c + logPred));")
            self._write("F_troe = pow(10., logFcent / (1.0 + troe*troe));")
            self._write("Corr[i] = F * F_troe;")
            self._outdent()
            self._write('}')

            self._outdent()
            self._write("}")

        if nsri > 0:
            self._write()
            self._write(self.line(" SRI"))
            self._write("{")
            self._indent()
            self._write("double alpha[%d];" % nsri)
            self._write("double redP, F, X, F_sri;")
            alpha_d = {}
            for i in range(isri[0],isri[1]):
                ii = i - isri[0]
                reaction = mechanism.reaction(id=i)
                if reaction.thirdBody:
                    alpha = self._enhancement(mechanism, reaction)
                    if alpha in alpha_d:
                        self._write("alpha[%d] = %s;" %(ii,alpha_d[alpha]))
                    else:
                        self._write("alpha[%d] = %s;" %(ii,alpha))
                        alpha_d[alpha] = "alpha[%d]" % ii

            if nsri >= 4:
                self._outdent()
                self._outdent()
                self._write('#ifdef __INTEL_COMPILER')
                self._indent()
                self._indent()
                self._write(' #pragma simd')
                self._outdent()
                self._outdent()
                self._write('#endif')
                self._indent()
                self._indent()
            self._write("for (int i=%d; i<%d; i++)" %(isri[0],isri[1]))
            self._write("{")
            self._indent()
            self._write("redP = alpha[i-%d] / k_f_save[i] * phase_units[i] * low_A[i] * exp(low_beta[i] * tc[0] - activation_units[i] * low_Ea[i] *invT);" % itroe[0])
            self._write("F = redP / (1.0 + redP);")
            self._write("logPred = log10(redP);")
            self._write("X = 1.0 / (1.0 + logPred*logPred);")
            self._write("F_sri = exp(X * log(sri_a[i] * exp(-sri_b[i]*invT)")
            self._write("   +  (sri_c[i] > 1.e-100 ? exp(T/sri_c[i]) : 0.0) )")
            self._write("   *  (sri_len[i] > 3 ? sri_d[i]*exp(sri_e[i]*tc[0]) : 1.0);")
            self._write("Corr[i] = F * F_sri;")
            self._outdent()
            self._write('}')

            self._outdent()
            self._write("}")

        if nlindemann > 0:
            self._write()
            self._write(self.line(" Lindemann"))
            self._write("{")
            self._indent()
            if nlindemann > 1:
                self._write("double alpha[%d];" % nlindemann)
            else:
                self._write("double alpha;")

            for i in range(ilindemann[0],ilindemann[1]):
                ii = i - ilindemann[0]
                reaction = mechanism.reaction(id=i)
                if reaction.thirdBody:
                    alpha = self._enhancement(mechanism, reaction)
                    if nlindemann > 1:
                        self._write("alpha[%d] = %s;" %(ii,alpha))
                    else:
                        self._write("alpha = %s;" %(alpha))

            if nlindemann == 1:
                self._write("double redP = alpha / k_f_save[%d] * phase_units[%d] * low_A[%d] * exp(low_beta[%d] * tc[0] - activation_units[%d] * low_Ea[%d] * invT);" 
                            % (ilindemann[0],ilindemann[0],ilindemann[0],ilindemann[0],ilindemann[0],ilindemann[0]))
                self._write("Corr[%d] = redP / (1. + redP);" % ilindemann[0])
            else:
                if nlindemann >= 4:
                    self._outdent()
                    self._write('#ifdef __INTEL_COMPILER')
                    self._indent()
                    self._write(' #pragma simd')
                    self._outdent()
                    self._write('#endif')
                    self._indent()
                self._write("for (int i=%d; i<%d; i++)" % (ilindemann[0], ilindemann[1]))
                self._write("{")
                self._indent()
                self._write("double redP = alpha[i-%d] / k_f_save[i] * phase_units[i] * low_A[i] * exp(low_beta[i] * tc[0] - activation_units[i] * low_Ea[i] * invT);"
                            % ilindemann[0])
                self._write("Corr[i] = redP / (1. + redP);")
                self._outdent()
                self._write('}')

            self._outdent()
            self._write("}")

        if n3body > 0:
            self._write()
            self._write(self.line(" simple three-body correction"))
            self._write("{")
            self._indent()
            self._write("double alpha;")
            alpha_save = ""
            for i in range(i3body[0],i3body[1]):
                reaction = mechanism.reaction(id=i)
                if reaction.thirdBody:
                    alpha = self._enhancement(mechanism, reaction)
                    if alpha != alpha_save:
                        alpha_save = alpha
                        self._write("alpha = %s;" % alpha)
                    self._write("Corr[%d] = alpha;" % i)
            self._outdent()
            self._write("}")

        self._write()
        self._write("for (int i=0; i<%d; i++)" % nclassd)
        self._write("{")
        self._indent()
        self._write("qf[i] *= Corr[i] * k_f_save[i];")
        self._write("qr[i] *= Corr[i] * k_f_save[i] / Kc_save[i];")
        self._outdent()
        self._write("}")
        
        if nspecial > 0:

            print "\n\n ***** WARNING: %d unclassified reactions\n" % nspecial

            self._write()
            self._write(self.line('unclassified reactions'))
            self._write('{')
            self._indent()

            self._write(self.line("reactions: %d to %d" % (ispecial[0]+1,ispecial[1])))

            #self._write('double Kc;                      ' + self.line('equilibrium constant'))
            self._write('double k_f;                     ' + self.line('forward reaction rate'))
            self._write('double k_r;                     ' + self.line('reverse reaction rate'))
            self._write('double q_f;                     ' + self.line('forward progress rate'))
            self._write('double q_r;                     ' + self.line('reverse progress rate'))
            self._write('double phi_f;                   '
                        + self.line('forward phase space factor'))
            self._write('double phi_r;                   ' + self.line('reverse phase space factor'))
            self._write('double alpha;                   ' + self.line('enhancement'))

            for i in range(ispecial[0],ispecial[1]):
                self._write()
                reaction = mechanism.reaction(id=i)
                self._write(self.line('reaction %d: %s' % (reaction.id, reaction.equation())))

                # compute the rates
                self._forwardRate(mechanism, reaction)
                self._reverseRate(mechanism, reaction)

                # store the progress rate
                self._write("qf[%d] = q_f;" % i)
                self._write("qr[%d] = q_r;" % i)

            self._outdent()
            self._write('}')

        self._write()
        self._write('return;')
        self._outdent()
        self._write('}')
        self._write("#endif")
        return


    def _DproductionRatePrecond(self, mechanism):

        species_list = [x.symbol for x in mechanism.species()]
        nSpecies = len(species_list)

        self._write()
        self._write(self.line('compute an approx to the reaction Jacobian (for preconditioning)'))
        self._write('AMREX_GPU_HOST_DEVICE void DWDOT_SIMPLIFIED(double *  J, double *  sc, double *  Tp, int * HP)')
        self._write('{')
        self._indent()

        self._write('double c[%d];' % (nSpecies))
        self._write()
        self._write('for (int k=0; k<%d; k++) {' % nSpecies)
        self._indent()
        self._write('c[k] = 1.e6 * sc[k];')
        self._outdent()
        self._write('}')

        self._write()
        self._write('aJacobian_precond(J, c, *Tp, *HP);')

        self._write()
        self._write('/* dwdot[k]/dT */')
        self._write('/* dTdot/d[X] */')
        self._write('for (int k=0; k<%d; k++) {' % nSpecies)
        self._indent()
        self._write('J[%d+k] *= 1.e-6;' % (nSpecies*(nSpecies+1)))
        self._write('J[k*%d+%d] *= 1.e6;' % (nSpecies+1, nSpecies))
        self._outdent()
        self._write('}')

        self._write()
        self._write('return;')
        self._outdent()
        self._write('}')
        return


    def _DproductionRate(self, mechanism):

        species_list = [x.symbol for x in mechanism.species()]
        nSpecies = len(species_list)

        self._write()
        self._write(self.line('compute the reaction Jacobian'))
        self._write('AMREX_GPU_HOST_DEVICE void DWDOT(double *  J, double *  sc, double *  Tp, int * consP)')
        self._write('{')
        self._indent()

        self._write('double c[%d];' % (nSpecies))
        self._write()
        self._write('for (int k=0; k<%d; k++) {' % nSpecies)
        self._indent()
        self._write('c[k] = 1.e6 * sc[k];')
        self._outdent()
        self._write('}')

        self._write()
        self._write('aJacobian(J, c, *Tp, *consP);')

        self._write()
        self._write('/* dwdot[k]/dT */')
        self._write('/* dTdot/d[X] */')
        self._write('for (int k=0; k<%d; k++) {' % nSpecies)
        self._indent()
        self._write('J[%d+k] *= 1.e-6;' % (nSpecies*(nSpecies+1)))
        self._write('J[k*%d+%d] *= 1.e6;' % (nSpecies+1, nSpecies))
        self._outdent()
        self._write('}')

        self._write()
        self._write('return;')
        self._outdent()
        self._write('}')
        return

    ################################
    # END MAIN PROD RATE GENERATORS
    ################################

    def _sparsity(self, mechanism):

        species_list = [x.symbol for x in mechanism.species()]
        nSpecies = len(species_list)

        ####
        self._write()
        self._write(self.line('compute the sparsity pattern of the chemistry Jacobian'))
        self._write('AMREX_GPU_HOST_DEVICE void SPARSITY_INFO( int * nJdata, int * consP, int NCELLS)')
        self._write('{')
        self._indent()

        self._write('double c[%d];' % (nSpecies))
        self._write('double J[%d];' % (nSpecies+1)**2)
        self._write()
        self._write('for (int k=0; k<%d; k++) {' % nSpecies)
        self._indent()
        self._write('c[k] = 1.0/ %f ;' % (nSpecies))
        self._outdent()
        self._write('}')

        self._write()
        self._write('aJacobian(J, c, 1500.0, *consP);')

        self._write()
        self._write('int nJdata_tmp = 0;')
        self._write('for (int k=0; k<%d; k++) {' % (nSpecies+1))
        self._indent()
        self._write('for (int l=0; l<%d; l++) {' % (nSpecies+1))
        self._indent()
        self._write('if(J[ %d * k + l] != 0.0){' % (nSpecies+1))
        self._indent()
        self._write('nJdata_tmp = nJdata_tmp + 1;')
        self._outdent()
        self._write('}')
        self._outdent()
        self._write('}')
        self._outdent()
        self._write('}')

        self._write()
        self._write('*nJdata = NCELLS * nJdata_tmp;')

        self._write()
        self._write('return;')
        self._outdent()

        self._write('}')
        self._write()
        self._write()

        ####
        self._write()
        self._write(self.line('compute the sparsity pattern of the system Jacobian'))
        self._write('AMREX_GPU_HOST_DEVICE void SPARSITY_INFO_SYST( int * nJdata, int * consP, int NCELLS)')
        self._write('{')
        self._indent()

        self._write('double c[%d];' % (nSpecies))
        self._write('double J[%d];' % (nSpecies+1)**2)
        self._write()
        self._write('for (int k=0; k<%d; k++) {' % nSpecies)
        self._indent()
        self._write('c[k] = 1.0/ %f ;' % (nSpecies))
        self._outdent()
        self._write('}')

        self._write()
        self._write('aJacobian(J, c, 1500.0, *consP);')

        self._write()
        self._write('int nJdata_tmp = 0;')
        self._write('for (int k=0; k<%d; k++) {' % (nSpecies+1))
        self._indent()
        self._write('for (int l=0; l<%d; l++) {' % (nSpecies+1))
        self._indent()
        self._write('if(k == l){')
        self._indent()
        self._write('nJdata_tmp = nJdata_tmp + 1;')
        self._outdent()
        self._write('} else {')
        self._indent()
        self._write('if(J[ %d * k + l] != 0.0){' % (nSpecies+1))
        self._indent()
        self._write('nJdata_tmp = nJdata_tmp + 1;')
        self._outdent()
        self._write('}')
        self._outdent()
        self._write('}')
        self._outdent()
        self._write('}')
        self._outdent()
        self._write('}')

        self._write()
        self._write('*nJdata = NCELLS * nJdata_tmp;')

        self._write()
        self._write('return;')
        self._outdent()

        self._write('}')
        self._write()
        self._write()

        ####
        self._write()
        self._write(self.line('compute the sparsity pattern of the simplified (for preconditioning) system Jacobian'))
        self._write('AMREX_GPU_HOST_DEVICE void SPARSITY_INFO_SYST_SIMPLIFIED( int * nJdata, int * consP)')
        self._write('{')
        self._indent()

        self._write('double c[%d];' % (nSpecies))
        self._write('double J[%d];' % (nSpecies+1)**2)
        self._write()
        self._write('for (int k=0; k<%d; k++) {' % nSpecies)
        self._indent()
        self._write('c[k] = 1.0/ %f ;' % (nSpecies))
        self._outdent()
        self._write('}')

        self._write()
        self._write('aJacobian_precond(J, c, 1500.0, *consP);')

        self._write()
        self._write('int nJdata_tmp = 0;')
        self._write('for (int k=0; k<%d; k++) {' % (nSpecies+1))
        self._indent()
        self._write('for (int l=0; l<%d; l++) {' % (nSpecies+1))
        self._indent()
        self._write('if(k == l){')
        self._indent()
        self._write('nJdata_tmp = nJdata_tmp + 1;')
        self._outdent()
        self._write('} else {')
        self._indent()
        self._write('if(J[ %d * k + l] != 0.0){' % (nSpecies+1))
        self._indent()
        self._write('nJdata_tmp = nJdata_tmp + 1;')
        self._outdent()
        self._write('}')
        self._outdent()
        self._write('}')
        self._outdent()
        self._write('}')
        self._outdent()
        self._write('}')

        self._write()
        self._write('nJdata[0] = nJdata_tmp;')

        self._write()
        self._write('return;')
        self._outdent()

        self._write('}')
        self._write()
        self._write()


        ####
        self._write(self.line('compute the sparsity pattern of the chemistry Jacobian in CSC format -- base 0'))
        self._write('AMREX_GPU_HOST_DEVICE void SPARSITY_PREPROC_CSC(int *  rowVals, int *  colPtrs, int * consP, int NCELLS)')
        self._write('{')
        self._indent()

        self._write('double c[%d];' % (nSpecies))
        self._write('double J[%d];' % (nSpecies+1)**2)
        self._write('int offset_row;')
        self._write('int offset_col;')
        self._write()
        self._write('for (int k=0; k<%d; k++) {' % nSpecies)
        self._indent()
        self._write('c[k] = 1.0/ %f ;' % (nSpecies))
        self._outdent()
        self._write('}')

        self._write()
        self._write('aJacobian(J, c, 1500.0, *consP);')

        self._write()
        self._write('colPtrs[0] = 0;')
        self._write('int nJdata_tmp = 0;')
        self._write('for (int nc=0; nc<NCELLS; nc++) {')
        self._indent()
        self._write('offset_row = nc * %d;' % (nSpecies+1))
        self._write('offset_col = nc * %d;' % (nSpecies+1))
        self._write('for (int k=0; k<%d; k++) {' % (nSpecies+1))
        self._indent()
        self._write('for (int l=0; l<%d; l++) {' % (nSpecies+1))
        self._indent()
        self._write('if(J[%d*k + l] != 0.0) {' % (nSpecies+1))
        self._indent()
        self._write('rowVals[nJdata_tmp] = l + offset_row; ')
        self._write('nJdata_tmp = nJdata_tmp + 1; ')
        self._outdent()
        self._write('}')
        self._outdent()
        self._write('}')
        self._write('colPtrs[offset_col + (k + 1)] = nJdata_tmp;')
        self._outdent()
        self._write('}')
        self._outdent()
        self._write('}')

        self._write()
        self._write('return;')
        self._outdent()

        self._write('}')
        self._write()

        ####
        self._write(self.line('compute the sparsity pattern of the chemistry Jacobian in CSR format -- base 0'))
        self._write('AMREX_GPU_HOST_DEVICE void SPARSITY_PREPROC_CSR(int * colVals, int * rowPtrs, int * consP, int NCELLS, int base)')
        self._write('{')
        self._indent()

        self._write('double c[%d];' % (nSpecies))
        self._write('double J[%d];' % (nSpecies+1)**2)
        self._write('int offset;')
        self._write()
        self._write('for (int k=0; k<%d; k++) {' % nSpecies)
        self._indent()
        self._write('c[k] = 1.0/ %f ;' % (nSpecies))
        self._outdent()
        self._write('}')

        self._write()
        self._write('aJacobian(J, c, 1500.0, *consP);')

        self._write()
        self._write('if (base == 1) {')
        self._indent()
        self._write('rowPtrs[0] = 1;')
        self._write('int nJdata_tmp = 1;')
        self._write('for (int nc=0; nc<NCELLS; nc++) {')
        self._indent()
        self._write('offset = nc * %d;' % (nSpecies+1))
        self._write('for (int l=0; l<%d; l++) {' % (nSpecies+1))
        self._indent()
        self._write('for (int k=0; k<%d; k++) {' % (nSpecies+1))
        self._indent()
        self._write('if(J[%d*k + l] != 0.0) {' % (nSpecies+1))
        self._indent()
        self._write('colVals[nJdata_tmp-1] = k+1 + offset; ')
        self._write('nJdata_tmp = nJdata_tmp + 1; ')
        self._outdent()
        self._write('}')
        self._outdent()
        self._write('}')
        self._write('rowPtrs[offset + (l + 1)] = nJdata_tmp;')
        self._outdent()
        self._write('}')
        self._outdent()
        self._write('}')
         
        self._outdent()
        self._write('} else {')
        self._indent()
        self._write('rowPtrs[0] = 0;')
        self._write('int nJdata_tmp = 0;')
        self._write('for (int nc=0; nc<NCELLS; nc++) {')
        self._indent()
        self._write('offset = nc * %d;' % (nSpecies+1))
        self._write('for (int l=0; l<%d; l++) {' % (nSpecies+1))
        self._indent()
        self._write('for (int k=0; k<%d; k++) {' % (nSpecies+1))
        self._indent()
        self._write('if(J[%d*k + l] != 0.0) {' % (nSpecies+1))
        self._indent()
        self._write('colVals[nJdata_tmp] = k + offset; ')
        self._write('nJdata_tmp = nJdata_tmp + 1; ')
        self._outdent()
        self._write('}')
        self._outdent()
        self._write('}')
        self._write('rowPtrs[offset + (l + 1)] = nJdata_tmp;')
        self._outdent()
        self._write('}')
        self._outdent()
        self._write('}')
        self._outdent()
        self._write('}')

        self._write()
        self._write('return;')
        self._outdent()

        self._write('}')
        self._write()

        ####
        self._write(self.line('compute the sparsity pattern of the system Jacobian'))
        self._write(self.line('CSR format BASE is user choice'))
        self._write('AMREX_GPU_HOST_DEVICE void SPARSITY_PREPROC_SYST_CSR(int * colVals, int * rowPtr, int * consP, int NCELLS, int base)')
        self._write('{')
        self._indent()

        self._write('double c[%d];' % (nSpecies))
        self._write('double J[%d];' % (nSpecies+1)**2)
        self._write('int offset;')

        self._write()
        self._write('for (int k=0; k<%d; k++) {' % nSpecies)
        self._indent()
        self._write('c[k] = 1.0/ %f ;' % (nSpecies))
        self._outdent()
        self._write('}')

        self._write()
        self._write('aJacobian(J, c, 1500.0, *consP);')

        self._write()
        self._write('if (base == 1) {')
        self._indent()
        self._write('rowPtr[0] = 1;')
        self._write('int nJdata_tmp = 1;')
        self._write('for (int nc=0; nc<NCELLS; nc++) {')
        self._indent()
        self._write('offset = nc * %d;' % (nSpecies+1));
        self._write('for (int l=0; l<%d; l++) {' % (nSpecies+1))
        self._indent()
        self._write('for (int k=0; k<%d; k++) {' % (nSpecies+1))
        self._indent()
        self._write('if (k == l) {')
        self._indent()
        self._write('colVals[nJdata_tmp-1] = l+1 + offset; ')
        self._write('nJdata_tmp = nJdata_tmp + 1; ')
        self._outdent()
        self._write('} else {')
        self._indent()
        self._write('if(J[%d*k + l] != 0.0) {' % (nSpecies+1))
        self._indent()
        self._write('colVals[nJdata_tmp-1] = k+1 + offset; ')
        self._write('nJdata_tmp = nJdata_tmp + 1; ')
        self._outdent()
        self._write('}')
        self._outdent()
        self._write('}')
        self._outdent()
        self._write('}')
        self._write('rowPtr[offset + (l + 1)] = nJdata_tmp;')
        self._outdent()
        self._write('}')
        self._outdent()
        self._write('}')
        self._outdent()
        self._write('} else {')
        self._indent()
        self._write('rowPtr[0] = 0;')
        self._write('int nJdata_tmp = 0;')
        self._write('for (int nc=0; nc<NCELLS; nc++) {')
        self._indent()
        self._write('offset = nc * %d;' % (nSpecies+1));
        self._write('for (int l=0; l<%d; l++) {' % (nSpecies+1))
        self._indent()
        self._write('for (int k=0; k<%d; k++) {' % (nSpecies+1))
        self._indent()
        self._write('if (k == l) {')
        self._indent()
        self._write('colVals[nJdata_tmp] = l + offset; ')
        self._write('nJdata_tmp = nJdata_tmp + 1; ')
        self._outdent()
        self._write('} else {')
        self._indent()
        self._write('if(J[%d*k + l] != 0.0) {' % (nSpecies+1))
        self._indent()
        self._write('colVals[nJdata_tmp] = k + offset; ')
        self._write('nJdata_tmp = nJdata_tmp + 1; ')
        self._outdent()
        self._write('}')
        self._outdent()
        self._write('}')
        self._outdent()
        self._write('}')
        self._write('rowPtr[offset + (l + 1)] = nJdata_tmp;')
        self._outdent()
        self._write('}')
        self._outdent()
        self._write('}')
        self._outdent()
        self._write('}')

        self._write()
        self._write('return;')
        self._outdent()

        self._write('}')
        self._write()

        ####
        self._write(self.line('compute the sparsity pattern of the simplified (for precond) system Jacobian on CPU'))
        self._write(self.line('BASE 0'))
        self._write('AMREX_GPU_HOST_DEVICE void SPARSITY_PREPROC_SYST_SIMPLIFIED_CSC(int * rowVals, int * colPtrs, int * indx, int * consP)')
        self._write('{')
        self._indent()

        self._write('double c[%d];' % (nSpecies))
        self._write('double J[%d];' % (nSpecies+1)**2)
        self._write()
        self._write('for (int k=0; k<%d; k++) {' % nSpecies)
        self._indent()
        self._write('c[k] = 1.0/ %f ;' % (nSpecies))
        self._outdent()
        self._write('}')

        self._write()
        self._write('aJacobian_precond(J, c, 1500.0, *consP);')

        self._write()
        self._write('colPtrs[0] = 0;')
        self._write('int nJdata_tmp = 0;')
        self._write('for (int k=0; k<%d; k++) {' % (nSpecies+1))
        self._indent()
        self._write('for (int l=0; l<%d; l++) {' % (nSpecies+1))
        self._indent()
        self._write('if (k == l) {')
        self._indent()
        self._write('rowVals[nJdata_tmp] = l; ')
        self._write('indx[nJdata_tmp] = %d*k + l;' % (nSpecies+1))
        self._write('nJdata_tmp = nJdata_tmp + 1; ')
        self._outdent()
        self._write('} else {')
        self._indent()
        self._write('if(J[%d*k + l] != 0.0) {' % (nSpecies+1))
        self._indent()
        self._write('rowVals[nJdata_tmp] = l; ')
        self._write('indx[nJdata_tmp] = %d*k + l;' % (nSpecies+1))
        self._write('nJdata_tmp = nJdata_tmp + 1; ')
        self._outdent()
        self._write('}')
        self._outdent()
        self._write('}')
        self._outdent()
        self._write('}')
        self._write('colPtrs[k+1] = nJdata_tmp;')
        self._outdent()
        self._write('}')

        self._write()
        self._write('return;')
        self._outdent()

        self._write('}')
        self._write()

        ####
        self._write(self.line('compute the sparsity pattern of the simplified (for precond) system Jacobian'))
        self._write(self.line('CSR format BASE is under choice'))
        self._write('AMREX_GPU_HOST_DEVICE void SPARSITY_PREPROC_SYST_SIMPLIFIED_CSR(int * colVals, int * rowPtr, int * consP, int base)')
        self._write('{')
        self._indent()

        self._write('double c[%d];' % (nSpecies))
        self._write('double J[%d];' % (nSpecies+1)**2)
        self._write()
        self._write('for (int k=0; k<%d; k++) {' % nSpecies)
        self._indent()
        self._write('c[k] = 1.0/ %f ;' % (nSpecies))
        self._outdent()
        self._write('}')

        self._write()
        self._write('aJacobian_precond(J, c, 1500.0, *consP);')

        self._write()
        self._write('if (base == 1) {')
        self._indent()
        self._write('rowPtr[0] = 1;')
        self._write('int nJdata_tmp = 1;')
        self._write('for (int l=0; l<%d; l++) {' % (nSpecies+1))
        self._indent()
        self._write('for (int k=0; k<%d; k++) {' % (nSpecies+1))
        self._indent()
        self._write('if (k == l) {')
        self._indent()
        self._write('colVals[nJdata_tmp-1] = l+1; ')
        self._write('nJdata_tmp = nJdata_tmp + 1; ')
        self._outdent()
        self._write('} else {')
        self._indent()
        self._write('if(J[%d*k + l] != 0.0) {' % (nSpecies+1))
        self._indent()
        self._write('colVals[nJdata_tmp-1] = k+1; ')
        self._write('nJdata_tmp = nJdata_tmp + 1; ')
        self._outdent()
        self._write('}')
        self._outdent()
        self._write('}')
        self._outdent()
        self._write('}')
        self._write('rowPtr[l+1] = nJdata_tmp;')
        self._outdent()
        self._write('}')
        self._outdent()
        self._write('} else {')
        self._indent()
        self._write('rowPtr[0] = 0;')
        self._write('int nJdata_tmp = 0;')
        self._write('for (int l=0; l<%d; l++) {' % (nSpecies+1))
        self._indent()
        self._write('for (int k=0; k<%d; k++) {' % (nSpecies+1))
        self._indent()
        self._write('if (k == l) {')
        self._indent()
        self._write('colVals[nJdata_tmp] = l; ')
        self._write('nJdata_tmp = nJdata_tmp + 1; ')
        self._outdent()
        self._write('} else {')
        self._indent()
        self._write('if(J[%d*k + l] != 0.0) {' % (nSpecies+1))
        self._indent()
        self._write('colVals[nJdata_tmp] = k; ')
        self._write('nJdata_tmp = nJdata_tmp + 1; ')
        self._outdent()
        self._write('}')
        self._outdent()
        self._write('}')
        self._outdent()
        self._write('}')
        self._write('rowPtr[l+1] = nJdata_tmp;')
        self._outdent()
        self._write('}')
        self._outdent()
        self._write('}')

        self._write()
        self._write('return;')
        self._outdent()
        self._write('}')
        self._write()
        return


    def _ajac_dummy(self, mechanism):

        nSpecies = len(mechanism.species())
        nReactions = len(mechanism.reaction())

        self._write()
        self._write('#ifndef AMREX_USE_CUDA')
        self._write(self.line('compute the reaction Jacobian on CPU'))
        self._write('void aJacobian(double *  J, double *  sc, double T, int consP)')
        self._write('{')
        self._indent()

        self._write('for (int i=0; i<%d; i++) {' % (nSpecies+1)**2)
        self._indent()
        self._write('J[i] = 0.0;')
        self._outdent()
        self._write('}')

        self._outdent()
        self._write('}')
        self._write('#endif')
        self._write()

        return


    def _ajacPrecond_dummy(self, mechanism):

        nSpecies = len(mechanism.species())
        nReactions = len(mechanism.reaction())

        self._write()
        self._write(self.line('compute an approx to the reaction Jacobian'))
        self._write('AMREX_GPU_HOST_DEVICE void aJacobian_precond(double *  J, double *  sc, double T, int HP)')
        self._write('{')
        self._indent()

        self._write('for (int i=0; i<%d; i++) {' % (nSpecies+1)**2)
        self._indent()
        self._write('J[i] = 0.0;')
        self._outdent()
        self._write('}')

        self._outdent()
        self._write('}')
        return
        

    ################################
    # THERMO #
    ################################

    def _dthermodT(self, mechanism):
        speciesInfo = self._analyzeThermodynamics(mechanism, 0)
        self._dcvpdT(speciesInfo)
        return


    def _analyzeThermodynamics(self, mechanism, QSS_Flag):
        lowT = 0.0
        highT = 1000000.0

        midpoints = {}

        lowT_qss = 0.0
        highT_qss = 1000000.0

        midpoints_qss = {}
        
        for species in mechanism.species():

            models = species.thermo
            if len(models) > 2:
                print 'species: ', species
                import pyre
                pyre.debug.Firewall.hit("unsupported configuration in species.thermo")
                return
            
            m1 = models[0]
            m2 = models[1]

            if m1.lowT < m2.lowT:
                lowRange = m1
                highRange = m2
            else:
                lowRange = m2
                highRange = m1

            low = lowRange.lowT
            mid = lowRange.highT
            high = highRange.highT

            if low > lowT:
                lowT = low
            if high < highT:
                highT = high

            midpoints.setdefault(mid, []).append((species, lowRange, highRange))
        
        self.lowT = lowT
        self.highT = highT

        for species in mechanism.qss_species():

            models = species.thermo
            if len(models) > 2:
                print 'species: ', species
                import pyre
                pyre.debug.Firewall.hit("unsupported configuration in species.thermo")
                return
            if len(models) == 0:
                return lowT_qss, highT_qss, midpoints_qss
            
            m1 = models[0]
            m2 = models[1]

            if m1.lowT < m2.lowT:
                lowRange = m1
                highRange = m2
            else:
                lowRange = m2
                highRange = m1

            low = lowRange.lowT
            mid = lowRange.highT
            high = highRange.highT

            if low > lowT_qss:
                lowT_qss = low
            if high < highT_qss:
                highT_qss = high

            midpoints_qss.setdefault(mid, []).append((species, lowRange, highRange))
        
        self.lowT_qss = lowT_qss
        self.highT_qss = highT_qss

        if QSS_Flag:
            return lowT_qss, highT_qss, midpoints_qss
        else:
            return lowT, highT, midpoints




    def _equilibriumConstants(self, mechanism):
        self._write()
        self._write()
        self._write(self.line('compute the equilibrium constants for each reaction'))
        self._write('void equilibriumConstants(double *  kc, double *  g_RT, double T)')
        self._write('{')
        self._indent()

        # compute the reference concentration
        self._write(self.line('reference concentration: P_atm / (RT) in inverse mol/m^3'))
        self._write('double refC = %g / %g / T;' % (atm.value, R.value))

        # compute the equilibrium constants
        for reaction in mechanism.reaction():
            self._write()
            self._write(self.line('reaction %d: %s' % (reaction.id, reaction.equation())))

            K_c = self._Kc(mechanism, reaction)
            self._write("kc[%d] = %s;" % (reaction.id - 1, K_c))

        self._write()
        self._write('return;')
        self._outdent()

        self._write('}')

        return


    def _Kc(self, mechanism, reaction):

        dim = 0
        dG = ""

        terms = []
        for symbol, coefficient in reaction.reactants:
            if coefficient == 1.0:
                factor = ""
            else:
                factor = "%f * " % coefficient

            if symbol in self.qss_list:
                terms.append("%sg_RT_qss[%d]" % (factor, mechanism.qss_species(symbol).id))
            else:
                terms.append("%sg_RT[%d]" % (factor, mechanism.species(symbol).id))

            dim -= coefficient
        dG += '(' + ' + '.join(terms) + ')'

        # flip the signs
        terms = []
        for symbol, coefficient in reaction.products:
            if coefficient == 1.0:
                factor = ""
            else:
                factor = "%f * " % coefficient

            if symbol in self.qss_list:
                terms.append("%sg_RT_qss[%d]" % (factor, mechanism.qss_species(symbol).id))
            else:
                terms.append("%sg_RT[%d]" % (factor, mechanism.species(symbol).id))

            dim += coefficient
        dG += ' - (' + ' + '.join(terms) + ')'

        K_p = 'exp(' + dG + ')'

        if dim == 0:
            conversion = ""
        elif dim > 0:
            if (dim == 1.0):
                conversion = "*".join(["refC"]) + ' * '
            else:
                conversion = "*".join(["pow(refC,%f)" % dim]) + ' * '
        else:
            if (dim == -1.0):
                conversion = "1.0 / (" + "*".join(["refC"]) + ') * '
            else:
                conversion = "1.0 / (" + "*".join(["pow(refC,%f)" % abs(dim)]) + ') * '

        K_c = conversion + K_p

        return K_c


    def _KcConv(self, mechanism, reaction):
        dim = 0
        for symbol, coefficient in reaction.reactants:
            dim -= coefficient
        # flip the signs
        for symbol, coefficient in reaction.products:
            dim += coefficient

        if dim == 0:
            conversion = ""
        elif dim > 0:
            if (dim == 1.0):
                conversion = "*".join(["refC"])
            else:
                conversion = "*".join(["pow(refC,%f)" % dim])
        else:
            if (dim == -1.0):
                conversion = "*".join(["refCinv"])
            else:
                conversion = "*".join(["pow(refCinv,%f)" % abs(dim)])

        return conversion


    def _sortedKcExpArg(self, mechanism, reaction):

        nSpecies = len(mechanism.species())+len(self.qss_species)

        terms = []
        # terms list filled with transported species first, referenced by index, and then QSS species at afterwards, referenced by their list order + nSpecies
        for i in range(nSpecies):
            terms.append('')
        for symbol, coefficient in reaction.reactants:
            if coefficient == 1.0:
                factor = " + "
            else:
                factor = " + %f*" % coefficient

            if symbol in self.qss_list:
                i = mechanism.qss_species(symbol).id
                terms[i+self.nSpecies] += "%sg_RT_qss[%d]"%(factor,i)
            else:
                i = mechanism.species(symbol).id
                terms[i] += "%sg_RT[%d]"%(factor,i)

        for symbol, coefficient in reaction.products:
            if coefficient == 1.0:
                factor = " - "    # flip the signs
            else:
                factor = " - %f*" % coefficient

            if symbol in self.qss_list:
                i = mechanism.qss_species(symbol).id
                terms[i+self.nSpecies] += "%sg_RT_qss[%d]"%(factor,i)
            else:
                i = mechanism.species(symbol).id
                terms[i] += "%sg_RT[%d]"%(factor,i)

        dG = ""
        for i in range(nSpecies):
            if terms[i]:
                dG += terms[i]
        if dG[0:3] == " + ":
            return dG[3:]
        else:
            return "-"+dG[3:]


    def _thermo_GPU(self, mechanism):
        speciesInfo = self._analyzeThermodynamics(mechanism, 0)
        QSSspeciesInfo = self._analyzeThermodynamics(mechanism, 1)
        
        self._gibbs_GPU(speciesInfo, 0)
        self._gibbs_GPU(QSSspeciesInfo, 1)
        self._helmholtz_GPU(speciesInfo)
        self._cv_GPU(speciesInfo)
        self._cp_GPU(speciesInfo)
        self._speciesInternalEnergy_GPU(speciesInfo)
        self._speciesEnthalpy_GPU(speciesInfo)
        self._speciesEntropy_GPU(speciesInfo)

        return


    def _dcvpdT(self, speciesInfo):

        self._write()
        self._write()
        self._write(self.line('compute d(Cp/R)/dT and d(Cv/R)/dT at the given temperature'))
        self._write(self.line('tc contains precomputed powers of T, tc[0] = log(T)'))
        self._generateThermoRoutine_GPU("dcvpRdT", self._dcpdTNASA, speciesInfo)

        return


    def _gibbs_GPU(self, speciesInfo, QSS_Flag):
        
        if QSS_Flag:
            name = "gibbs_qss"
        else:
            name = "gibbs"
            
        self._write()
        self._write()
        self._write(self.line('compute the g/(RT) at the given temperature'))
        self._write(self.line('tc contains precomputed powers of T, tc[0] = log(T)'))
        self._generateThermoRoutine_GPU(name, self._gibbsNASA, speciesInfo, 1)

        return


    def _helmholtz_GPU(self, speciesInfo):

        self._write()
        self._write()
        self._write(self.line('compute the a/(RT) at the given temperature'))
        self._write(self.line('tc contains precomputed powers of T, tc[0] = log(T)'))
        self._generateThermoRoutine_GPU("helmholtz", self._helmholtzNASA, speciesInfo, 1)

        return



    def _cv_GPU(self, speciesInfo):

        self._write()
        self._write()
        self._write(self.line('compute Cv/R at the given temperature'))
        self._write(self.line('tc contains precomputed powers of T, tc[0] = log(T)'))
        self._generateThermoRoutine_GPU("cv_R", self._cvNASA, speciesInfo)

        return
    
    def _cp_GPU(self, speciesInfo):

        self._write()
        self._write()
        self._write(self.line('compute Cp/R at the given temperature'))
        self._write(self.line('tc contains precomputed powers of T, tc[0] = log(T)'))
        self._generateThermoRoutine_GPU("cp_R", self._cpNASA, speciesInfo)

        return


    def _speciesInternalEnergy_GPU(self, speciesInfo):

        self._write()
        self._write()
        self._write(self.line('compute the e/(RT) at the given temperature'))
        self._write(self.line('tc contains precomputed powers of T, tc[0] = log(T)'))
        self._generateThermoRoutine_GPU("speciesInternalEnergy", self._internalEnergy, speciesInfo, 1)

        return


    def _speciesEnthalpy_GPU(self, speciesInfo):

        self._write()
        self._write()
        self._write(self.line('compute the h/(RT) at the given temperature (Eq 20)'))
        self._write(self.line('tc contains precomputed powers of T, tc[0] = log(T)'))
        self._generateThermoRoutine_GPU("speciesEnthalpy", self._enthalpyNASA, speciesInfo, 1)

        return


    def _speciesEntropy_GPU(self, speciesInfo):

        self._write()
        self._write()
        self._write(self.line('compute the S/R at the given temperature (Eq 21)'))
        self._write(self.line('tc contains precomputed powers of T, tc[0] = log(T)'))
        self._generateThermoRoutine_GPU("speciesEntropy", self._entropyNASA, speciesInfo)

        return


    def _generateThermoRoutine_GPU(self, name, expressionGenerator, speciesInfo, needsInvT=0):

        lowT, highT, midpoints = speciesInfo

        #print("SPECIES INFO FROM generateThermoRoutine_GPU: ")
        # print(speciesInfo)
        
        self._write('AMREX_GPU_HOST_DEVICE void %s(double * species, double *  tc)' % name)
        self._write('{')

        self._indent()

        # declarations
        self._write()
        self._write(self.line('temperature'))
        self._write('double T = tc[1];')
        if needsInvT != 0:
           self._write('double invT = 1 / T;')
        if needsInvT == 2:
           self._write('double invT2 = invT*invT;')

        # temperature check
        # self._write()
        # self._write(self.line('check the temperature value'))
        # self._write('if (T < %g || T > %g) {' % (lowT, highT))
        # self._indent()
        # self._write(
        #     'fprintf(stderr, "temperature %%g is outside the range (%g, %g)", T);'
        #     % (lowT, highT))
        # self._write('return;')
        # self._outdent()
        # self._write('}')
                    
        for midT, speciesList in midpoints.items():

            self._write('')
            self._write(self.line('species with midpoint at T=%g kelvin' % midT))
            self._write('if (T < %g) {' % midT)
            self._indent()

            for species, lowRange, highRange in speciesList:
                self._write(self.line('species %d: %s' % (species.id, species.symbol)))
                self._write('species[%d] =' % species.id)
                self._indent()
                expressionGenerator(lowRange.parameters)
                self._outdent()

            self._outdent()
            self._write('} else {')
            self._indent()

            for species, lowRange, highRange in speciesList:
                self._write(self.line('species %d: %s' % (species.id, species.symbol)))
                self._write('species[%d] =' % species.id)
                self._indent()
                expressionGenerator(highRange.parameters)
                self._outdent()

            self._outdent()
            self._write('}')
            
        self._write('return;')
        self._outdent()

        self._write('}')

        return


    def _cpNASA(self, parameters):
        self._write('%+15.8e' % parameters[0])
        self._write('%+15.8e * tc[1]' % parameters[1])
        self._write('%+15.8e * tc[2]' % parameters[2])
        self._write('%+15.8e * tc[3]' % parameters[3])
        self._write('%+15.8e * tc[4];' % parameters[4])
        return

    def _dcpdTNASA(self, parameters):
        self._write('%+15.8e' % parameters[1])
        self._write('%+15.8e * tc[1]' % (parameters[2]*2.))
        self._write('%+15.8e * tc[2]' % (parameters[3]*3.))
        self._write('%+15.8e * tc[3];' % (parameters[4]*4.))
        return

    def _cvNASA(self, parameters):
        self._write('%+15.8e' % (parameters[0] - 1.0))
        self._write('%+15.8e * tc[1]' % parameters[1])
        self._write('%+15.8e * tc[2]' % parameters[2])
        self._write('%+15.8e * tc[3]' % parameters[3])
        self._write('%+15.8e * tc[4];' % parameters[4])
        return

    def _enthalpyNASA(self, parameters):
        self._write('%+15.8e' % parameters[0])
        self._write('%+15.8e * tc[1]' % (parameters[1]/2))
        self._write('%+15.8e * tc[2]' % (parameters[2]/3))
        self._write('%+15.8e * tc[3]' % (parameters[3]/4))
        self._write('%+15.8e * tc[4]' % (parameters[4]/5))
        self._write('%+15.8e * invT;' % (parameters[5]))
        return

    def _internalEnergy(self, parameters):
        self._write('%+15.8e' % (parameters[0] - 1.0))
        self._write('%+15.8e * tc[1]' % (parameters[1]/2))
        self._write('%+15.8e * tc[2]' % (parameters[2]/3))
        self._write('%+15.8e * tc[3]' % (parameters[3]/4))
        self._write('%+15.8e * tc[4]' % (parameters[4]/5))
        self._write('%+15.8e * invT;' % (parameters[5]))
        return

    def _gibbsNASA(self, parameters):
        self._write('%+20.15e * invT' % parameters[5])
        self._write('%+20.15e' % (parameters[0] - parameters[6]))
        self._write('%+20.15e * tc[0]' % (-parameters[0]))
        self._write('%+20.15e * tc[1]' % (-parameters[1]/2))
        self._write('%+20.15e * tc[2]' % (-parameters[2]/6))
        self._write('%+20.15e * tc[3]' % (-parameters[3]/12))
        self._write('%+20.15e * tc[4];' % (-parameters[4]/20))
        return
    
    def _helmholtzNASA(self, parameters):
        self._write('%+15.8e * invT' % parameters[5])
        self._write('%+15.8e' % (parameters[0] - parameters[6] - 1.0))
        self._write('%+15.8e * tc[0]' % (-parameters[0]))
        self._write('%+15.8e * tc[1]' % (-parameters[1]/2))
        self._write('%+15.8e * tc[2]' % (-parameters[2]/6))
        self._write('%+15.8e * tc[3]' % (-parameters[3]/12))
        self._write('%+15.8e * tc[4];' % (-parameters[4]/20))
        return

    def _entropyNASA(self, parameters):
        self._write('%+15.8e * tc[0]' % parameters[0])
        self._write('%+15.8e * tc[1]' % (parameters[1]))
        self._write('%+15.8e * tc[2]' % (parameters[2]/2))
        self._write('%+15.8e * tc[3]' % (parameters[3]/3))
        self._write('%+15.8e * tc[4]' % (parameters[4]/4))
        self._write('%+15.8e ;' % (parameters[6]))
        return

    ################################
    # END THERMO #
    ################################

    def _atomicWeight(self, mechanism):

        self._write()
        self._write()
        self._write(self.line('save atomic weights into array'))
        self._write('void atomicWeight(double *  awt)')
        self._write('{')
        self._indent()
        import pyre
        periodic = pyre.handbook.periodicTable()
        for element in mechanism.element():
            aw = mechanism.element(element.symbol).weight
            if not aw:
                aw = periodic.symbol(element.symbol.capitalize()).atomicWeight

            self._write('awt[%d] = %f; ' % (
                element.id, aw) + self.line('%s' % element.symbol))

        self._write()
        self._write('return;')
        self._outdent()

        self._write('}')
        self._write()

        return 


    def _T_given_ey(self, mechanism):
        self._write()
        self._write(self.line(' get temperature given internal energy in mass units and mass fracs'))
        self._write('AMREX_GPU_HOST_DEVICE void GET_T_GIVEN_EY(double *  e, double *  y, double *  t, int * ierr)')
        self._write('{')
        self._write('#ifdef CONVERGENCE')
        self._indent()
        self._write('const int maxiter = 5000;')
        self._write('const double tol  = 1.e-12;')
        self._outdent()
        self._write('#else')
        self._indent()
        self._write('const int maxiter = 200;')
        self._write('const double tol  = 1.e-6;')
        self._outdent()
        self._write('#endif')
        self._indent()
        self._write('double ein  = *e;')
        self._write('double tmin = 90;'+self.line('max lower bound for thermo def'))
        self._write('double tmax = 4000;'+self.line('min upper bound for thermo def'))
        self._write('double e1,emin,emax,cv,t1,dt;')
        self._write('int i;'+self.line(' loop counter'))
        self._write('CKUBMS(&tmin, y, &emin);')
        self._write('CKUBMS(&tmax, y, &emax);')
        self._write('if (ein < emin) {')
        self._indent()
        self._write(self.line('Linear Extrapolation below tmin'))
        self._write('CKCVBS(&tmin, y, &cv);')
        self._write('*t = tmin - (emin-ein)/cv;')
        self._write('*ierr = 1;')
        self._write('return;')
        self._outdent()
        self._write('}')
        self._write('if (ein > emax) {')
        self._indent()
        self._write(self.line('Linear Extrapolation above tmax'))
        self._write('CKCVBS(&tmax, y, &cv);')
        self._write('*t = tmax - (emax-ein)/cv;')
        self._write('*ierr = 1;')
        self._write('return;')
        self._outdent()
        self._write('}')
        self._write('t1 = *t;')
        self._write('if (t1 < tmin || t1 > tmax) {')
        self._indent()
        self._write('t1 = tmin + (tmax-tmin)/(emax-emin)*(ein-emin);')
        self._outdent()
        self._write('}')
        self._write('for (i = 0; i < maxiter; ++i) {')
        self._indent()
        self._write('CKUBMS(&t1,y,&e1);')
        self._write('CKCVBS(&t1,y,&cv);')
        self._write('dt = (ein - e1) / cv;')
        self._write('if (dt > 100.) { dt = 100.; }')
        self._write('else if (dt < -100.) { dt = -100.; }')
        self._write('else if (fabs(dt) < tol) break;')
        self._write('else if (t1+dt == t1) break;')
        self._write('t1 += dt;')
        self._outdent()
        self._write('}')
        self._write('*t = t1;')
        self._write('*ierr = 0;')
        self._write('return;')
        self._outdent()
        self._write('}')
        self._write()

    def _T_given_hy(self, mechanism):
        self._write(self.line(' get temperature given enthalpy in mass units and mass fracs'))
        self._write('AMREX_GPU_HOST_DEVICE void GET_T_GIVEN_HY(double *  h, double *  y, double *  t, int * ierr)')
        self._write('{')
        self._write('#ifdef CONVERGENCE')
        self._indent()
        self._write('const int maxiter = 5000;')
        self._write('const double tol  = 1.e-12;')
        self._outdent()
        self._write('#else')
        self._indent()
        self._write('const int maxiter = 200;')
        self._write('const double tol  = 1.e-6;')
        self._outdent()
        self._write('#endif')
        self._indent()
        self._write('double hin  = *h;')
        self._write('double tmin = 90;'+self.line('max lower bound for thermo def'))
        self._write('double tmax = 4000;'+self.line('min upper bound for thermo def'))
        self._write('double h1,hmin,hmax,cp,t1,dt;')
        self._write('int i;'+self.line(' loop counter'))
        self._write('CKHBMS(&tmin, y, &hmin);')
        self._write('CKHBMS(&tmax, y, &hmax);')
        self._write('if (hin < hmin) {')
        self._indent()
        self._write(self.line('Linear Extrapolation below tmin'))
        self._write('CKCPBS(&tmin, y, &cp);')
        self._write('*t = tmin - (hmin-hin)/cp;')
        self._write('*ierr = 1;')
        self._write('return;')
        self._outdent()
        self._write('}')
        self._write('if (hin > hmax) {')
        self._indent()
        self._write(self.line('Linear Extrapolation above tmax'))
        self._write('CKCPBS(&tmax, y, &cp);')
        self._write('*t = tmax - (hmax-hin)/cp;')
        self._write('*ierr = 1;')
        self._write('return;')
        self._outdent()
        self._write('}')
        self._write('t1 = *t;')
        self._write('if (t1 < tmin || t1 > tmax) {')
        self._indent()
        self._write('t1 = tmin + (tmax-tmin)/(hmax-hmin)*(hin-hmin);')
        self._outdent()
        self._write('}')
        self._write('for (i = 0; i < maxiter; ++i) {')
        self._indent()
        self._write('CKHBMS(&t1,y,&h1);')
        self._write('CKCPBS(&t1,y,&cp);')
        self._write('dt = (hin - h1) / cp;')
        self._write('if (dt > 100.) { dt = 100.; }')
        self._write('else if (dt < -100.) { dt = -100.; }')
        self._write('else if (fabs(dt) < tol) break;')
        self._write('else if (t1+dt == t1) break;')
        self._write('t1 += dt;')
        self._outdent()
        self._write('}')
        self._write('*t = t1;')
        self._write('*ierr = 0;')
        self._write('return;')
        self._outdent()
        self._write('}')


    ################################
    # TRANSPORT #
    ################################
       
    def _transport(self, mechanism):
        speciesTransport = self._analyzeTransport(mechanism)
        NLITE=0
        idxLightSpecs = []
        for spec in self.species:
            if spec.weight < 5.0:
                NLITE+=1
                idxLightSpecs.append(spec.id)
        self._miscTransInfo(KK=self.nSpecies, NLITE=NLITE, do_declarations=False)
        self._wt(False)
        self._eps(mechanism, speciesTransport, False)
        self._sig(mechanism, speciesTransport, False)
        self._dip(mechanism, speciesTransport, False)
        self._pol(mechanism, speciesTransport, False)
        self._zrot(mechanism, speciesTransport, False)
        self._nlin(mechanism, speciesTransport, False)

        self._viscosity(speciesTransport, False, NTFit=50)
        self._diffcoefs(speciesTransport, False, NTFit=50)
        self._lightSpecs(idxLightSpecs, False)
        self._thermaldiffratios(speciesTransport, idxLightSpecs, False, NTFit=50)

        return


    def _wt(self, do_declarations):

        self._write()
        self._write()
        self._write(self.line('the molecular weights in g/mol'))

        if (do_declarations):
            self._write('#if defined(BL_FORT_USE_UPPERCASE)')
            self._write('#define egtransetWT EGTRANSETWT')
            self._write('#elif defined(BL_FORT_USE_LOWERCASE)')
            self._write('#define egtransetWT egtransetwt')
            self._write('#elif defined(BL_FORT_USE_UNDERSCORE)')
            self._write('#define egtransetWT egtransetwt_')
            self._write('#endif')

        self._write('void %s(double* %s ) {' % ("egtransetWT", "WT"))
        self._indent()

        for species in self.species:
            self._write('%s[%d] = %.8E;' % ("WT", species.id, float(species.weight)))

        self._outdent()
        self._write('}')
        return


    def _eps(self, mechanism, speciesTransport, do_declarations):
        self._write()
        self._write()
        self._write(self.line('the lennard-jones potential well depth eps/kb in K'))
        #i=0
        #expression=[]
        #for species in mechanism.species():
        #    expression[i] = float(species.trans[0].eps)
        #    i++
        self._generateTransRoutineSimple(mechanism, ["egtransetEPS", "EGTRANSETEPS", "egtranseteps", "egtranseteps_", "EPS"], 1, speciesTransport, do_declarations)

        return
    

    def _sig(self, mechanism, speciesTransport, do_declarations):
        self._write()
        self._write()
        self._write(self.line('the lennard-jones collision diameter in Angstroms'))
        self._generateTransRoutineSimple(mechanism, ["egtransetSIG", "EGTRANSETSIG", "egtransetsig", "egtransetsig_", "SIG"], 2, speciesTransport, do_declarations)

        return


    def _dip(self, mechanism, speciesTransport, do_declarations):
        self._write()
        self._write()
        self._write(self.line('the dipole moment in Debye'))
        self._generateTransRoutineSimple(mechanism, ["egtransetDIP", "EGTRANSETDIP", "egtransetdip", "egtransetdip_", "DIP"], 3, speciesTransport, do_declarations)

        return


    def _pol(self, mechanism, speciesTransport, do_declarations):
        self._write()
        self._write()
        self._write(self.line('the polarizability in cubic Angstroms'))
        self._generateTransRoutineSimple(mechanism, ["egtransetPOL", "EGTRANSETPOL", "egtransetpol", "egtransetpol_", "POL"], 4, speciesTransport, do_declarations)

        return


    def _zrot(self, mechanism, speciesTransport, do_declarations):
        self._write()
        self._write()
        self._write(self.line('the rotational relaxation collision number at 298 K'))
        self._generateTransRoutineSimple(mechanism, ["egtransetZROT", "EGTRANSETZROT", "egtransetzrot", "egtransetzrot_", "ZROT"], 5, speciesTransport, do_declarations)

        return


    def _nlin(self, mechanism, speciesTransport, do_declarations):
        self._write()
        self._write()
        self._write(self.line('0: monoatomic, 1: linear, 2: nonlinear'))
        #self._generateTransRoutineSimple(["egtransetNLIN", "EGTRANSETNLIN", "egtransetNLIN", "egtransetNLIN_", "NLIN"], 0, speciesTransport)

        if (do_declarations):
            self._write('#if defined(BL_FORT_USE_UPPERCASE)')
            self._write('#define egtransetNLIN EGTRANSETNLIN')
            self._write('#elif defined(BL_FORT_USE_LOWERCASE)')
            self._write('#define egtransetNLIN egtransetnlin')
            self._write('#elif defined(BL_FORT_USE_UNDERSCORE)')
            self._write('#define egtransetNLIN egtransetnlin_')
            self._write('#endif')

        self._write('void egtransetNLIN(int* NLIN) {')
        self._indent()

        for species in mechanism.species():
            self._write('%s[%d] = %d;' % ("NLIN", species.id, int(speciesTransport[species][0])))

        self._outdent()
        self._write('}')

        return


    def _viscosity(self, speciesTransport, do_declarations, NTFit):
        #compute single constants in g/cm/s
        kb = 1.3806503e-16
        Na = 6.02214199e23 
        RU = 8.31447e7
        #conversion coefs
        AtoCM = 1.0e-8
        DEBYEtoCGS = 1.0e-18
        #temperature increment  
        dt = (self.highT-self.lowT) / (NTFit-1)
        #factor dependent upon the molecule
        m_crot = np.zeros(self.nSpecies)
        m_cvib = np.zeros(self.nSpecies)
        isatm = np.zeros(self.nSpecies)
        for spec in speciesTransport:
            if int(speciesTransport[spec][0]) == 0:
                m_crot[spec.id] = 0.0
                m_cvib[spec.id] = 0.0
                isatm[spec.id] = 0.0
            elif int(speciesTransport[spec][0]) == 1:
                m_crot[spec.id] = 1.0
                m_cvib[spec.id] = 5.0 / 2.0
                isatm[spec.id] = 1.0
            else:
                m_crot[spec.id] = 1.5
                m_cvib[spec.id] = 3.0
                isatm[spec.id] = 1.0
        #viscosities coefs (4 per spec)
        cofeta = {}
        #conductivities coefs (4 per spec)
        coflam = {}
        for spec in speciesTransport:
            spvisc = []
            spcond = []
            tlog = []
            for n in range(NTFit):
                t = self.lowT + dt*n
                #variables
                #eq. (2)
                tr = t/ float(speciesTransport[spec][1])
                conversion = DEBYEtoCGS * DEBYEtoCGS / AtoCM / AtoCM / AtoCM / kb 
                dst = 0.5 * conversion * float(speciesTransport[spec][3])**2 / (float(speciesTransport[spec][1]) \
                        * float(speciesTransport[spec][2])**3)
                #viscosity of spec at t
                #eq. (1)
                conversion = AtoCM * AtoCM
                visc = (5.0 / 16.0) * np.sqrt(np.pi * self.species[spec.id].weight * kb * t / Na) / \
                    (self.om22_CHEMKIN(tr,dst) * np.pi * \
                    float(speciesTransport[spec][2]) * float(speciesTransport[spec][2]) * conversion)
                #conductivity of spec at t
                #eq. (30)
                conversion = AtoCM * AtoCM
                m_red = self.species[spec.id].weight / (2.0 * Na)
                diffcoef = (3.0 / 16.0) * np.sqrt(2.0 * np.pi * kb**3 * t**3 / m_red) /  \
                        (10.0 * np.pi * self.om11_CHEMKIN(tr,dst) * float(speciesTransport[spec][2]) * \
                        float(speciesTransport[spec][2]) * conversion)
                #eq. (19)
                cv_vib_R = (self._getCVdRspecies(t, spec) - m_cvib[spec.id]) * isatm[spec.id]
                rho_atm = 10.0 * self.species[spec.id].weight /(RU * t)
                f_vib = rho_atm * diffcoef / visc
                #eq. (20)
                A = 2.5 - f_vib
                #eqs. (21) + (32-33)
                cv_rot_R = m_crot[spec.id]
                #note: the T corr is not applied in CANTERA
                B = (float(speciesTransport[spec][5]) \
                        * self.Fcorr(298.0, float(speciesTransport[spec][1])) / self.Fcorr(t, float(speciesTransport[spec][1])) \
                        + (2.0 / np.pi) * ((5.0 / 3.0 ) * cv_rot_R  + f_vib))
                #eq. (18)
                f_rot = f_vib * (1.0 + 2.0 / np.pi * A / B )
                #eq. (17) 
                cv_trans_R = 3.0 / 2.0 
                f_trans = 5.0 / 2.0 * (1.0 - 2.0 / np.pi * A / B * cv_rot_R / cv_trans_R )
                if (int(speciesTransport[spec][0]) == 0):
                    cond = (visc * RU / self.species[spec.id].weight) * \
                            (5.0 / 2.0) * cv_trans_R
                else:
                    cond = (visc * RU / self.species[spec.id].weight) * \
                        (f_trans * cv_trans_R + f_rot * cv_rot_R + f_vib * cv_vib_R)

                #log transformation for polyfit
                tlog.append(np.log(t))
                spvisc.append(np.log(visc))
                spcond.append(np.log(cond))

            cofeta[spec.id] = np.polyfit(tlog, spvisc, 3)
            coflam[spec.id] = np.polyfit(tlog, spcond, 3)

        #header for visco
        self._write()
        self._write()
        self._write(self.line('Poly fits for the viscosities, dim NO*KK'))
        if (do_declarations):
            self._write('#if defined(BL_FORT_USE_UPPERCASE)')
            self._write('#define egtransetCOFETA EGTRANSETCOFETA')
            self._write('#elif defined(BL_FORT_USE_LOWERCASE)')
            self._write('#define egtransetCOFETA egtransetcofeta')
            self._write('#elif defined(BL_FORT_USE_UNDERSCORE)')
            self._write('#define egtransetCOFETA egtransetcofeta_')
            self._write('#endif')

        #visco coefs
        self._write('void egtransetCOFETA(double* COFETA) {')
        self._indent()

        for spec in self.species:
            for i in range(4):
                self._write('%s[%d] = %.8E;' % ('COFETA', spec.id*4+i, cofeta[spec.id][3-i]))

        self._outdent()
        self._write('}')

        #header for cond
        self._write()
        self._write()
        self._write(self.line('Poly fits for the conductivities, dim NO*KK'))
        if (do_declarations):
            self._write('#if defined(BL_FORT_USE_UPPERCASE)')
            self._write('#define egtransetCOFLAM EGTRANSETCOFLAM')
            self._write('#elif defined(BL_FORT_USE_LOWERCASE)')
            self._write('#define egtransetCOFLAM egtransetcoflam')
            self._write('#elif defined(BL_FORT_USE_UNDERSCORE)')
            self._write('#define egtransetCOFLAM egtransetcoflam_')
            self._write('#endif')

        #visco coefs
        self._write('void egtransetCOFLAM(double* COFLAM) {')

        self._indent()

        for spec in self.species:
            for i in range(4):
                self._write('%s[%d] = %.8E;' % ('COFLAM', spec.id*4+i, coflam[spec.id][3-i]))

        self._outdent()

        self._write('}')

        return


    def _diffcoefs(self, speciesTransport, do_declarations, NTFit) :

        #REORDERING OF SPECS
        specOrdered = []
        for i in range(self.nSpecies):
            for spec in speciesTransport:
                if spec.id == i:
                    specOrdered.append(spec)
                    break
        #checks
        #for spec in speciesTransport:
        #    print spec.symbol, spec.id
        #for i in range(self.nSpecies):
        #    print i, specOrdered[i].id, specOrdered[i].symbol
        #stop

        #compute single constants in g/cm/s
        kb = 1.3806503e-16
        Na = 6.02214199e23 
        #conversion coefs
        AtoCM = 1.0e-8
        DEBYEtoCGS = 1.0e-18
        PATM = 0.1013250000000000E+07
        #temperature increment  
        dt = (self.highT-self.lowT) / (NTFit-1)
        #diff coefs (4 per spec pair) 
        cofd = []
        for i,spec1 in enumerate(specOrdered):
            cofd.append([])
            if (i != spec1.id):
                print "Problem in _diffcoefs computation"
                stop
            for j,spec2 in enumerate(specOrdered[0:i+1]):
                if (j != spec2.id):
                    print "Problem in _diffcoefs computation"
                    stop
                #eq. (9)
                sigm = (0.5 * (float(speciesTransport[spec1][2]) + float(speciesTransport[spec2][2])) * AtoCM)\
                        * self.Xi(spec1, spec2, speciesTransport)**(1.0/6.0)
                #eq. (4)
                m_red = self.species[spec1.id].weight * self.species[spec2.id].weight / \
                        (self.species[spec1.id].weight + self.species[spec2.id].weight) / Na
                #eq. (8) & (14)
                epsm_k = np.sqrt(float(speciesTransport[spec1][1]) * float(speciesTransport[spec2][1])) \
                        * self.Xi(spec1, spec2, speciesTransport)**2.0

                #eq. (15)
                conversion = DEBYEtoCGS * DEBYEtoCGS / kb  
                dst = 0.5 * conversion * float(speciesTransport[spec1][3]) * float(speciesTransport[spec2][3]) / \
                    (epsm_k * sigm**3)
                if self.Xi_bool(spec1, spec2, speciesTransport)==False:
                    dst = 0.0
                #enter the loop on temperature
                spdiffcoef = []
                tlog = []
                for n in range(NTFit):
                   t = self.lowT + dt*n
                   tr = t/ epsm_k
                   #eq. (3)
                   #note: these are "corrected" in CHEMKIN not in CANTERA... we chose not to
                   difcoeff = 3.0 / 16.0 * 1 / PATM * (np.sqrt(2.0 * np.pi * t**3 * kb**3 / m_red) / \
                           ( np.pi * sigm * sigm * self.om11_CHEMKIN(tr,dst)))

                   #log transformation for polyfit
                   tlog.append(np.log(t))
                   spdiffcoef.append(np.log(difcoeff))

                cofd[i].append(np.polyfit(tlog, spdiffcoef, 3))

        #use the symmetry for upper triangular terms 
        #note: starting with this would be preferable (only one bigger loop)
        #note2: or write stuff differently !
        #for i,spec1 in enumerate(specOrdered):
        #    for j,spec2 in enumerate(specOrdered[i+1:]):
        #        cofd[i].append(cofd[spec2.id][spec1.id])

        #header for diffusion coefs
        self._write()
        self._write()
        self._write(self.line('Poly fits for the diffusion coefficients, dim NO*KK*KK'))
        if (do_declarations):
            self._write('#if defined(BL_FORT_USE_UPPERCASE)')
            self._write('#define egtransetCOFD EGTRANSETCOFD')
            self._write('#elif defined(BL_FORT_USE_LOWERCASE)')
            self._write('#define egtransetCOFD egtransetcofd')
            self._write('#elif defined(BL_FORT_USE_UNDERSCORE)')
            self._write('#define egtransetCOFD egtransetcofd_')
            self._write('#endif')

        #coefs
        self._write('void egtransetCOFD(double* COFD) {')

        self._indent()

        for i,spec1 in enumerate(specOrdered):
            #for j,spec2 in enumerate(specOrdered):
            for j,spec2 in enumerate(specOrdered[0:i+1]):
                for k in range(4):
                    #self._write('%s[%d] = %.8E;' % ('COFD', i*self.nSpecies*4+j*4+k, cofd[j][i][3-k]))
                    self._write('%s[%d] = %.8E;' % ('COFD', i*self.nSpecies*4+j*4+k, cofd[i][j][3-k]))
            for j,spec2 in enumerate(specOrdered[i+1:]):
                for k in range(4):
                    self._write('%s[%d] = %.8E;' % ('COFD', i*self.nSpecies*4+(j+i+1)*4+k, cofd[j+i+1][i][3-k]))

        self._outdent()

        self._write('}')
        
        return


    def _lightSpecs(self, speclist, do_declarations):
        #header 
        self._write()
        self._write()
        self._write(self.line('List of specs with small weight, dim NLITE'))
        if (do_declarations):
            self._write('#if defined(BL_FORT_USE_UPPERCASE)')
            self._write('#define egtransetKTDIF EGTRANSETKTDIF')
            self._write('#elif defined(BL_FORT_USE_LOWERCASE)')
            self._write('#define egtransetKTDIF egtransetktdif')
            self._write('#elif defined(BL_FORT_USE_UNDERSCORE)')
            self._write('#define egtransetKTDIF egtransetktdif_')
            self._write('#endif')

        #coefs
        self._write('void egtransetKTDIF(int* KTDIF) {')
        self._indent()

        for i in range(len(speclist)):
            self._write('%s[%d] = %d;' % ('KTDIF', i, speclist[i]+1))

        self._outdent()
        self._write('}')
        
        return


    def _thermaldiffratios(self, speciesTransport, lightSpecList, do_declarations, NTFit):

        # This is an overhaul of CHEMKIN version III
        #REORDERING OF SPECS
        specOrdered = []
        for i in range(self.nSpecies):
            for spec in speciesTransport:
                if spec.id == i:
                    specOrdered.append(spec)
                    break

        #compute single constants in g/cm/s
        kb = 1.3806503e-16
        #conversion coefs
        DEBYEtoCGS = 1.0e-18
        AtoCM = 1.0e-8
        #temperature increment  
        dt = (self.highT-self.lowT) / (NTFit-1)
        #diff ratios (4 per spec pair involving light species) 
        coftd = []
        k = -1
        for i,spec1 in enumerate(specOrdered):
            if (i != spec1.id):
                print "Problem in _thermaldiffratios computation"
                stop
            if spec1.id in lightSpecList:
                k = k + 1
                if (lightSpecList[k] != spec1.id):
                    print "Problem in  _thermaldiffratios computation"
                    stop
                coftd.append([])
                epsi = float(speciesTransport[spec1][1]) * kb
                sigi = float(speciesTransport[spec1][2]) * AtoCM
                poli = float(speciesTransport[spec1][4]) * AtoCM * AtoCM * AtoCM
                #eq. (12)
                poliRed = poli / sigi**3
                for j,spec2 in enumerate(specOrdered):
                    if (j != spec2.id):
                        print "Problem in _thermaldiffratios computation"
                        stop
                    #eq. (53)
                    Wji = (self.species[spec2.id].weight - self.species[spec1.id].weight) / \
                            (self.species[spec1.id].weight + self.species[spec2.id].weight) 
                    epsj = float(speciesTransport[spec2][1]) * kb
                    sigj = float(speciesTransport[spec2][2]) * AtoCM
                    dipj = float(speciesTransport[spec2][3]) * DEBYEtoCGS
                    #eq. (13)
                    dipjRed = dipj / np.sqrt(epsj*sigj**3)
                    epsRatio = epsj / epsi
                    tse = 1.0 + 0.25*poliRed*dipjRed**2*np.sqrt(epsRatio)
                    eok = tse**2 * np.sqrt(float(speciesTransport[spec1][1]) * float(speciesTransport[spec2][1]))
                    #enter the loop on temperature
                    spthdiffcoef = []
                    tTab = []
                    for n in range(NTFit):
                       t = self.lowT + dt*n
                       tslog = np.log(t) - np.log(eok)
                       #eq. (53)
                       thdifcoeff = 15.0 / 2.0 * Wji * (2.0 * self.astar(tslog) + 5.0) * (6.0 * self.cstar(tslog) - 5.0) / \
                               (self.astar(tslog) * (16.0 * self.astar(tslog) - 12.0 * self.bstar(tslog) + 55.0))

                       #log transformation for polyfit
                       tTab.append(t)
                       spthdiffcoef.append(thdifcoeff)

                    coftd[k].append(np.polyfit(tTab, spthdiffcoef, 3))

        #header for thermal diff ratios
        self._write()
        self._write()
        self._write(self.line('Poly fits for thermal diff ratios, dim NO*NLITE*KK'))
        if (do_declarations):
            self._write('#if defined(BL_FORT_USE_UPPERCASE)')
            self._write('#define egtransetCOFTD EGTRANSETCOFTD')
            self._write('#elif defined(BL_FORT_USE_LOWERCASE)')
            self._write('#define egtransetCOFTD egtransetcoftd')
            self._write('#elif defined(BL_FORT_USE_UNDERSCORE)')
            self._write('#define egtransetCOFTD egtransetcoftd_')
            self._write('#endif')

        #visco coefs
        self._write('void egtransetCOFTD(double* COFTD) {')
        self._indent()

        for i in range(len(coftd)):
            for j in range(self.nSpecies):
                for k in range(4):
                    self._write('%s[%d] = %.8E;' % ('COFTD', i*4*self.nSpecies+j*4+k, coftd[i][j][3-k]))

        self._outdent()
        self._write('}')

        return


    def _analyzeTransport(self, mechanism):

        transdata = {}

        for species in mechanism.species():

            models = species.trans
            if len(models) > 2:
                print 'species: ', species
                import pyre
                pyre.debug.Firewall.hit("unsupported configuration in species.trans")
                return
            
            m1 = models[0]

            lin = m1.parameters[0]
            eps = m1.eps
            sig = m1.sig
            dip = m1.dip
            pol = m1.pol
            zrot = m1.zrot

            transdata[species] = [lin, eps, sig, dip, pol, zrot]

        return transdata


    def _miscTransInfo(self, KK, NLITE, do_declarations, NO=4):

        self._write()
        self._write()
        LENIMC = 4*KK+NLITE
        self._generateTransRoutineInteger(["egtransetLENIMC", "EGTRANSETLENIMC", "egtransetlenimc", "egtransetlenimc_", "LENIMC"], LENIMC, do_declarations)

        self._write()
        self._write()
        LENRMC = (19+2*NO+NO*NLITE)*KK+(15+NO)*KK**2
        self._generateTransRoutineInteger(["egtransetLENRMC", "EGTRANSETLENRMC", "egtransetlenrmc", "egtransetlenrmc_", "LENRMC"], LENRMC, do_declarations)

        self._write()
        self._write()
        self._generateTransRoutineInteger(["egtransetNO", "EGTRANSETNO", "egtransetno", "egtransetno_", "NO"], NO, do_declarations)

        self._write()
        self._write()
        self._generateTransRoutineInteger(["egtransetKK", "EGTRANSETKK", "egtransetkk", "egtransetkk_", "KK"], KK, do_declarations)

        self._write()
        self._write()
        self._generateTransRoutineInteger(["egtransetNLITE", "EGTRANSETNLITE", "egtransetnlite", "egtransetnlite_", "NLITE"], NLITE, do_declarations)

        self._write()
        self._write()
        self._write(self.line('Patm in ergs/cm3'))

        if (do_declarations):
            self._write('#if defined(BL_FORT_USE_UPPERCASE)')
            self._write('#define egtransetPATM EGTRANSETPATM')
            self._write('#elif defined(BL_FORT_USE_LOWERCASE)')
            self._write('#define egtransetPATM egtransetpatm')
            self._write('#elif defined(BL_FORT_USE_UNDERSCORE)')
            self._write('#define egtransetPATM egtransetpatm_')
            self._write('#endif')

        self._write('void egtransetPATM(double* PATM) {')
        self._indent()
        self._write('*PATM =   0.1013250000000000E+07;}')
        self._outdent()

        return


    def _generateTransRoutineSimple(self, mechanism, nametab, id, speciesTransport, do_declarations):

        if (do_declarations):
            self._write('#if defined(BL_FORT_USE_UPPERCASE)')
            self._write('#define %s %s' % (nametab[0], nametab[1]))
            self._write('#elif defined(BL_FORT_USE_LOWERCASE)')
            self._write('#define %s %s' % (nametab[0], nametab[2]))
            self._write('#elif defined(BL_FORT_USE_UNDERSCORE)')
            self._write('#define %s %s' % (nametab[0], nametab[3]))
            self._write('#endif')

        self._write('void %s(double* %s ) {' % (nametab[0], nametab[4]))
        self._indent()

        for species in mechanism.species():
            self._write('%s[%d] = %.8E;' % (nametab[4], species.id, float(speciesTransport[species][id])))

        self._outdent()
        self._write('}')

        return


    def _generateTransRoutineInteger(self, nametab, expression, do_declarations):

        if (do_declarations):
            self._write('#if defined(BL_FORT_USE_UPPERCASE)')
            self._write('#define %s %s' % (nametab[0], nametab[1]))
            self._write('#elif defined(BL_FORT_USE_LOWERCASE)')
            self._write('#define %s %s' % (nametab[0], nametab[2]))
            self._write('#elif defined(BL_FORT_USE_UNDERSCORE)')
            self._write('#define %s %s' % (nametab[0], nametab[3]))
            self._write('#endif')

        self._write('void %s(int* %s ) {' % (nametab[0], nametab[4]))
        self._indent()

        self._write('*%s = %d;}' % (nametab[4], expression ))
        self._outdent()

        return


    def astar(self, tslog):

        aTab = [.1106910525E+01, -.7065517161E-02,-.1671975393E-01,
                .1188708609E-01,  .7569367323E-03,-.1313998345E-02,
                .1720853282E-03]

        B = aTab[6]
        for i in range(6):
            B = aTab[5-i] + B*tslog

        return B


    def bstar(self, tslog):

        bTab = [.1199673577E+01, -.1140928763E+00,-.2147636665E-02,
                .2512965407E-01, -.3030372973E-02,-.1445009039E-02,
                .2492954809E-03]

        B = bTab[6]
        for i in range(6):
            B = bTab[5-i] + B*tslog

        return B


    def cstar(self, tslog):

        cTab = [ .8386993788E+00,  .4748325276E-01, .3250097527E-01,
                -.1625859588E-01, -.2260153363E-02, .1844922811E-02,
                -.2115417788E-03]

        B = cTab[6]
        for i in range(6):
            B = cTab[5-i] + B*tslog

        return B


    def Xi(self, spec1, spec2, speciesTransport):

        dipmin = 1e-20
        #1 is polar, 2 is nonpolar
        #err in eq. (11) ?
        if (float(speciesTransport[spec2][3]) < dipmin) and (float(speciesTransport[spec1][3]) > dipmin):
            xi = 1.0 + 1.0/4.0 * self.redPol(spec2, speciesTransport)*self.redDip(spec1, speciesTransport) *\
                    self.redDip(spec1, speciesTransport) *\
                    np.sqrt(float(speciesTransport[spec1][1]) / float(speciesTransport[spec2][1]))
        #1 is nonpolar, 2 is polar
        elif (float(speciesTransport[spec2][3]) > dipmin) and (float(speciesTransport[spec1][3]) < dipmin):
            xi = 1.0 + 1.0/4.0 * self.redPol(spec1, speciesTransport)*self.redDip(spec2, speciesTransport) *\
                    self.redDip(spec2, speciesTransport) *\
                    np.sqrt(float(speciesTransport[spec2][1]) / float(speciesTransport[spec1][1]))
        #normal case, either both polar or both nonpolar
        else:
            xi = 1.0

        return xi


    def Xi_bool(self, spec1, spec2, speciesTransport):

        dipmin = 1e-20
        #1 is polar, 2 is nonpolar
        #err in eq. (11) ?
        if (float(speciesTransport[spec2][3]) < dipmin) and (float(speciesTransport[spec1][3]) > dipmin):
            xi_b = False
        #1 is nonpolar, 2 is polar
        elif (float(speciesTransport[spec2][3]) > dipmin) and (float(speciesTransport[spec1][3]) < dipmin):
            xi_b = False
        #normal case, either both polar or both nonpolar
        else:
            xi_b = True

        return xi_b


    def redPol(self, spec, speciesTransport): 

        return (float(speciesTransport[spec][4]) / float(speciesTransport[spec][2])**3.0)


    def redDip(self, spec, speciesTransport): 

        #compute single constants in g/cm/s
        kb = 1.3806503e-16
        #conversion coefs
        AtoCM = 1.0e-8
        DEBYEtoCGS = 1.0e-18
        convert = DEBYEtoCGS / np.sqrt( kb * AtoCM**3.0 )
        return convert * float(speciesTransport[spec][3]) / \
                np.sqrt(float(speciesTransport[spec][1]) * float(speciesTransport[spec][2])**3.0)


    def Fcorr(self, t, eps_k):

        thtwo = 3.0 / 2.0
        return 1 + np.pi**(thtwo) / 2.0 * np.sqrt(eps_k / t) + \
                (np.pi**2 / 4.0 + 2.0) * (eps_k / t) + \
                (np.pi * eps_k / t)**(thtwo)


    def om11(self, tr, dst):

        # This is an overhaul of CANTERA version 2.3
        #range of dst
        dstTab = [0.0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 2.5]

        #range of tr
        trTab = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0,
                5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0,
                18.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 75.0, 100.0]

        #tab of astar corresp. to (tr, dst)
        #CANTERA
        astarTab = [1.0065, 1.0840, 1.0840, 1.0840, 1.0840, 1.0840, 1.0840, 1.0840,
                1.0231, 1.0660, 1.0380, 1.0400, 1.0430, 1.0500, 1.0520, 1.0510,
                1.0424, 1.0450, 1.0480, 1.0520, 1.0560, 1.0650, 1.0660, 1.0640,
                1.0719, 1.0670, 1.0600, 1.0550, 1.0580, 1.0680, 1.0710, 1.0710,
                1.0936, 1.0870, 1.0770, 1.0690, 1.0680, 1.0750, 1.0780, 1.0780,
                1.1053, 1.0980, 1.0880, 1.0800, 1.0780, 1.0820, 1.0840, 1.0840,
                1.1104, 1.1040, 1.0960, 1.0890, 1.0860, 1.0890, 1.0900, 1.0900,
                1.1114, 1.1070, 1.1000, 1.0950, 1.0930, 1.0950, 1.0960, 1.0950,
                1.1104, 1.1070, 1.1020, 1.0990, 1.0980, 1.1000, 1.1000, 1.0990,
                1.1086, 1.1060, 1.1020, 1.1010, 1.1010, 1.1050, 1.1050, 1.1040,
                1.1063, 1.1040, 1.1030, 1.1030, 1.1040, 1.1080, 1.1090, 1.1080,
                1.1020, 1.1020, 1.1030, 1.1050, 1.1070, 1.1120, 1.1150, 1.1150,
                1.0985, 1.0990, 1.1010, 1.1040, 1.1080, 1.1150, 1.1190, 1.1200,
                1.0960, 1.0960, 1.0990, 1.1030, 1.1080, 1.1160, 1.1210, 1.1240,
                1.0943, 1.0950, 1.0990, 1.1020, 1.1080, 1.1170, 1.1230, 1.1260,
                1.0934, 1.0940, 1.0970, 1.1020, 1.1070, 1.1160, 1.1230, 1.1280,
                1.0926, 1.0940, 1.0970, 1.0990, 1.1050, 1.1150, 1.1230, 1.1300,
                1.0934, 1.0950, 1.0970, 1.0990, 1.1040, 1.1130, 1.1220, 1.1290,
                1.0948, 1.0960, 1.0980, 1.1000, 1.1030, 1.1120, 1.1190, 1.1270,
                1.0965, 1.0970, 1.0990, 1.1010, 1.1040, 1.1100, 1.1180, 1.1260,
                1.0997, 1.1000, 1.1010, 1.1020, 1.1050, 1.1100, 1.1160, 1.1230,
                1.1025, 1.1030, 1.1040, 1.1050, 1.1060, 1.1100, 1.1150, 1.1210,
                1.1050, 1.1050, 1.1060, 1.1070, 1.1080, 1.1110, 1.1150, 1.1200,
                1.1072, 1.1070, 1.1080, 1.1080, 1.1090, 1.1120, 1.1150, 1.1190,
                1.1091, 1.1090, 1.1090, 1.1100, 1.1110, 1.1130, 1.1150, 1.1190,
                1.1107, 1.1110, 1.1110, 1.1110, 1.1120, 1.1140, 1.1160, 1.1190,
                1.1133, 1.1140, 1.1130, 1.1140, 1.1140, 1.1150, 1.1170, 1.1190,
                1.1154, 1.1150, 1.1160, 1.1160, 1.1160, 1.1170, 1.1180, 1.1200,
                1.1172, 1.1170, 1.1170, 1.1180, 1.1180, 1.1180, 1.1190, 1.1200,
                1.1186, 1.1190, 1.1190, 1.1190, 1.1190, 1.1190, 1.1200, 1.1210,
                1.1199, 1.1200, 1.1200, 1.1200, 1.1200, 1.1210, 1.1210, 1.1220,
                1.1223, 1.1220, 1.1220, 1.1220, 1.1220, 1.1230, 1.1230, 1.1240,
                1.1243, 1.1240, 1.1240, 1.1240, 1.1240, 1.1240, 1.1250, 1.1250,
                1.1259, 1.1260, 1.1260, 1.1260, 1.1260, 1.1260, 1.1260, 1.1260,
                1.1273, 1.1270, 1.1270, 1.1270, 1.1270, 1.1270, 1.1270, 1.1280,
                1.1297, 1.1300, 1.1300, 1.1300, 1.1300, 1.1300, 1.1300, 1.1290,
                1.1339, 1.1340, 1.1340, 1.1350, 1.1350, 1.1340, 1.1340, 1.1320,
                1.1364, 1.1370, 1.1370, 1.1380, 1.1390, 1.1380, 1.1370, 1.1350,
                1.14187, 1.14187, 1.14187, 1.14187, 1.14187, 1.14187, 1.14187,
                1.14187]


        #Find for each fixed tr the poly of deg 6 in dst approx astar values
        #store the poly coefs in m_apoly
        m_apoly = []
        for i in range(37):
            dstDeg = 6
            #Polynomial coefficients, highest power first 
            polycoefs = np.polyfit(dstTab,astarTab[8*(i+1):8*(i+2)],dstDeg)
            m_apoly.append(polycoefs)

        #Find 3 referenced temp points around tr
        for i in range(37):
            if tr<trTab[i]:
                break
        i1 = max(i-1, 0)
        i2 = i1+3
        if (i2 > 36):
            i2 = 36
            i1 = i2 - 3
        #compute astar value for these 3 points
        values = []
        for j in range(i1,i2):
            if (dst == 0.0):
                values.append(astarTab[8*(j+1)])
            else:
                poly6 = np.poly1d(m_apoly[j])
                values.append(poly6(dst))

        #interpolate to find real tr value
        trTab_log = []
        for j in range(len(trTab)):
            trTab_log.append(np.log(trTab[j]))

        astar_interp = self.quadInterp(np.log(tr), trTab_log[i1:i2], values)

        return self.om22(tr,dst)/astar_interp


    def om11_CHEMKIN(self, tr, dst):

        # This is an overhaul of CANTERA version 2.3
        #range of dst
        dstTab = [0.0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 2.5]

        #range of tr
        trTab = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0,
                5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0,
                18.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 75.0, 100.0]

        #tab of omega11 corresp. to (tr, dst)
        #CANTERA
        omegaTab = [4.008, 4.002, 4.655, 5.52, 6.454, 8.214, 9.824, 11.31,
                3.130 , 3.164 , 3.355 , 3.721 , 4.198 , 5.23  , 6.225 , 7.160,
                2.649 , 2.657 , 2.77  , 3.002 , 3.319 , 4.054 , 4.785 , 5.483 ,
                2.314 , 2.32  , 2.402 , 2.572 , 2.812 , 3.386 , 3.972 , 4.539 ,
                2.066 , 2.073 , 2.14  , 2.278 , 2.472 , 2.946 , 3.437 , 3.918 ,
                1.877 , 1.885 , 1.944 , 2.06  , 2.225 , 2.628 , 3.054 , 3.747 ,
                1.729 , 1.738 , 1.79  , 1.893 , 2.036 , 2.388 , 2.763 , 3.137 ,
                1.6122, 1.622 , 1.67  , 1.76  , 1.886 , 2.198 , 2.535 , 2.872 ,
                1.517 , 1.527 , 1.572 , 1.653 , 1.765 , 2.044 , 2.35  , 2.657 ,
                1.44  , 1.45  , 1.49  , 1.564 , 1.665 , 1.917 , 2.196 , 2.4780,
                1.3204, 1.33  , 1.364 , 1.425 , 1.51  , 1.72  , 1.956 , 2.199,
                1.234 , 1.24  , 1.272 , 1.324 , 1.394 , 1.573 , 1.777 , 1.99,
                1.168 , 1.176 , 1.202 , 1.246 , 1.306 , 1.46  , 1.64  , 1.827,
                1.1166, 1.124 , 1.146 , 1.185 , 1.237 , 1.372 , 1.53  , 1.7,
                1.075 , 1.082 , 1.102 , 1.135 , 1.181 , 1.3   , 1.441 , 1.592,
                1.0006, 1.005 , 1.02  , 1.046 , 1.08  , 1.17  , 1.278 , 1.397,
                 .95  ,  .9538,  .9656,  .9852, 1.012 , 1.082 , 1.168 , 1.265,
                 .9131,  .9162,  .9256,  .9413,  .9626, 1.019 , 1.09  , 1.17,
                 .8845,  .8871,  .8948,  .9076,  .9252,  .972 , 1.03  , 1.098,
                 .8428,  .8446,  .850 ,  .859 ,  .8716,  .9053,  .9483,  .9984,
                 .813 ,  .8142,  .8183,  .825 ,  .8344,  .8598,  .8927,  .9316,
                 .7898,  .791 ,  .794 ,  .7993,  .8066,  .8265,  .8526,  .8836,
                 .7711,  .772 ,  .7745,  .7788,  .7846,  .8007,  .822 ,  .8474,
                 .7555,  .7562,  .7584,  .7619,  .7667,  .78  ,  .7976,  .8189,
                 .7422,  .743 ,  .7446,  .7475,  .7515,  .7627,  .7776,  .796 ,
                 .72022, .7206,  .722 ,  .7241,  .7271,  .7354,  .7464,  .76  ,
                 .7025,  .703 ,  .704 ,  .7055,  .7078,  .7142,  .7228,  .7334,
                 .68776, .688,   .6888,  .6901,  .6919,  .697 ,  .704 ,  .7125,
                 .6751,  .6753,  .676 ,  .677 ,  .6785,  .6827,  .6884,  .6955,
                 .664 ,  .6642,  .6648,  .6657,  .6669,  .6704,  .6752,  .681,
                 .6414,  .6415,  .6418,  .6425,  .6433,  .6457,  .649 ,  .653,
                 .6235,  .6236,  .6239,  .6243,  .6249,  .6267,  .629 ,  .632,
                 .60882, .6089,  .6091,  .6094,  .61  ,  .6112,  .613 ,  .6154,
                 .5964,  .5964,  .5966,  .597 ,  .5972,  .5983,  .600 ,  .6017,
                 .5763,  .5763,  .5764,  .5766,  .5768,  .5775,  .5785,  .58,
                 .5415,  .5415,  .5416,  .5416,  .5418,  .542 ,  .5424,  .543,
                 .518 ,  .518 ,  .5182,  .5184,  .5184,  .5185,  .5186,  .5187]

        #First test on tr
        if (tr > 75.0):
            omeg12 = 0.623 - 0.136e-2*tr + 0.346e-5*tr*tr - 0.343e-8*tr*tr*tr
        else:
            #Find tr idx in trTab
            if (tr <= 0.2):
                ii = 1
            else:
                ii = 36
            for i in range(1,37):
                if (tr > trTab[i-1]) and (tr <= trTab[i]):
                    ii = i
                    break
            #Find dst idx in dstTab 
            if (abs(dst) >= 1.0e-5):
                if (dst <= 0.25):
                    kk = 1
                else:
                    kk = 6
                for i in range(1,7):
                    if (dstTab[i-1] < dst) and (dstTab[i] >= dst):
                        kk = i
                        break
                #Find surrounding values and interpolate
                #First on dst
                vert = np.zeros(3) 
                for i in range(3):
                    arg = np.zeros(3) 
                    val = np.zeros(3) 
                    for k in range(3):
                      arg[k] = dstTab[kk-1+k]
                      val[k] = omegaTab[8*(ii-1+i) + (kk-1+k)]
                    vert[i] = self.qinterp(dst, arg, val)
                #Second on tr
                arg = np.zeros(3) 
                for i in range(3):
                   arg[i] = trTab[ii-1+i]
                omeg12 = self.qinterp(tr, arg, vert)
            else:
                arg = np.zeros(3) 
                val = np.zeros(3) 
                for i in range(3):
                   arg[i] = trTab[ii-1+i]
                   val[i] = omegaTab[8*(ii-1+i)]
                omeg12 =self. qinterp(tr, arg, val)

        return omeg12


    def om22_CHEMKIN(self, tr, dst):

        # This is an overhaul of CANTERA version 2.3
        #range of dst
        dstTab = [0.0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 2.5]

        #range of tr
        trTab = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0,
                5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0,
                18.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 75.0, 100.0]

        #tab of omega22 corresp. to (tr, dst)
        #CANTERA
        omegaTab = [4.1005, 4.266,  4.833,  5.742,  6.729,  8.624,  10.34,  11.89,
                3.2626, 3.305,  3.516,  3.914,  4.433,  5.57,   6.637,  7.618,
                2.8399, 2.836,  2.936,  3.168,  3.511,  4.329,  5.126,  5.874,
                2.531,  2.522,  2.586,  2.749,  3.004,  3.64,   4.282,  4.895,
                2.2837, 2.277,  2.329,  2.46,   2.665,  3.187,  3.727,  4.249,
                2.0838, 2.081,  2.13,   2.243,  2.417,  2.862,  3.329,  3.786,
                1.922,  1.924,  1.97,   2.072,  2.225,  2.614,  3.028,  3.435,
                1.7902, 1.795,  1.84,   1.934,  2.07,   2.417,  2.788,  3.156,
                1.6823, 1.689,  1.733,  1.82,   1.944,  2.258,  2.596,  2.933,
                1.5929, 1.601,  1.644,  1.725,  1.838,  2.124,  2.435,  2.746,
                1.4551, 1.465,  1.504,  1.574,  1.67,   1.913,  2.181,  2.451,
                1.3551, 1.365,  1.4,    1.461,  1.544,  1.754,  1.989,  2.228,
                1.28,   1.289,  1.321,  1.374,  1.447,  1.63,   1.838,  2.053,
                1.2219, 1.231,  1.259,  1.306,  1.37,   1.532,  1.718,  1.912,
                1.1757, 1.184,  1.209,  1.251,  1.307,  1.451,  1.618,  1.795,
                1.0933, 1.1,    1.119,  1.15,   1.193,  1.304,  1.435,  1.578,
                1.0388, 1.044,  1.059,  1.083,  1.117,  1.204,  1.31,   1.428,
                0.99963, 1.004, 1.016,  1.035,  1.062,  1.133,  1.22,   1.319,
                0.96988, 0.9732, 0.983,  0.9991, 1.021,  1.079,  1.153,  1.236,
                0.92676, 0.9291, 0.936,  0.9473, 0.9628, 1.005,  1.058,  1.121,
                0.89616, 0.8979, 0.903,  0.9114, 0.923,  0.9545, 0.9955, 1.044,
                0.87272, 0.8741, 0.878,  0.8845, 0.8935, 0.9181, 0.9505, 0.9893,
                0.85379, 0.8549, 0.858,  0.8632, 0.8703, 0.8901, 0.9164, 0.9482,
                0.83795, 0.8388, 0.8414, 0.8456, 0.8515, 0.8678, 0.8895, 0.916,
                0.82435, 0.8251, 0.8273, 0.8308, 0.8356, 0.8493, 0.8676, 0.8901,
                0.80184, 0.8024, 0.8039, 0.8065, 0.8101, 0.8201, 0.8337, 0.8504,
                0.78363, 0.784,  0.7852, 0.7872, 0.7899, 0.7976, 0.8081, 0.8212,
                0.76834, 0.7687, 0.7696, 0.7712, 0.7733, 0.7794, 0.7878, 0.7983,
                0.75518, 0.7554, 0.7562, 0.7575, 0.7592, 0.7642, 0.7711, 0.7797,
                0.74364, 0.7438, 0.7445, 0.7455, 0.747,  0.7512, 0.7569, 0.7642,
                0.71982, 0.72,   0.7204, 0.7211, 0.7221, 0.725,  0.7289, 0.7339,
                0.70097, 0.7011, 0.7014, 0.7019, 0.7026, 0.7047, 0.7076, 0.7112,
                0.68545, 0.6855, 0.6858, 0.6861, 0.6867, 0.6883, 0.6905, 0.6932,
                0.67232, 0.6724, 0.6726, 0.6728, 0.6733, 0.6743, 0.6762, 0.6784,
                0.65099, 0.651,  0.6512, 0.6513, 0.6516, 0.6524, 0.6534, 0.6546,
                0.61397, 0.6141, 0.6143, 0.6145, 0.6147, 0.6148, 0.6148, 0.6147,
                0.5887, 0.5889, 0.5894, 0.59,   0.5903, 0.5901, 0.5895, 0.5885]

        #First test on tr
        if (tr > 75.0):
            omeg12 = 0.703 - 0.146e-2*tr + 0.357e-5*tr*tr - 0.343e-8*tr*tr*tr
        else:
            #Find tr idx in trTab
            if (tr <= 0.2):
                ii = 1
            else:
                ii = 36
            for i in range(1,37):
                if (tr > trTab[i-1]) and (tr <= trTab[i]):
                    ii = i
                    break
            #Find dst idx in dstTab 
            if (abs(dst) >= 1.0e-5):
                if (dst <= 0.25):
                    kk = 1
                else:
                    kk = 6
                for i in range(1,7):
                    if (dstTab[i-1] < dst) and (dstTab[i] >= dst):
                        kk = i
                        break
                #Find surrounding values and interpolate
                #First on dst
                vert = np.zeros(3) 
                for i in range(3):
                    arg = np.zeros(3) 
                    val = np.zeros(3) 
                    for k in range(3):
                      arg[k] = dstTab[kk-1+k]
                      val[k] = omegaTab[8*(ii-1+i) + (kk-1+k)]
                    vert[i] = self.qinterp(dst, arg, val)
                #Second on tr
                arg = np.zeros(3) 
                for i in range(3):
                   arg[i] = trTab[ii-1+i]
                omeg12 = self.qinterp(tr, arg, vert)
            else:
                arg = np.zeros(3) 
                val = np.zeros(3) 
                for i in range(3):
                   arg[i] = trTab[ii-1+i]
                   val[i] = omegaTab[8*(ii-1+i)]
                omeg12 =self. qinterp(tr, arg, val)

        return omeg12


    def om22(self, tr, dst):

        # This is an overhaul of CANTERA version 2.3
        #range of dst
        dstTab = [0.0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 2.5]

        #range of tr
        trTab = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0,
                5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0,
                18.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 75.0, 100.0]

        #tab of omega22 corresp. to (tr, dst)
        #CANTERA
        omegaTab = [4.1005, 4.266,  4.833,  5.742,  6.729,  8.624,  10.34,  11.89,
                3.2626, 3.305,  3.516,  3.914,  4.433,  5.57,   6.637,  7.618,
                2.8399, 2.836,  2.936,  3.168,  3.511,  4.329,  5.126,  5.874,
                2.531,  2.522,  2.586,  2.749,  3.004,  3.64,   4.282,  4.895,
                2.2837, 2.277,  2.329,  2.46,   2.665,  3.187,  3.727,  4.249,
                2.0838, 2.081,  2.13,   2.243,  2.417,  2.862,  3.329,  3.786,
                1.922,  1.924,  1.97,   2.072,  2.225,  2.614,  3.028,  3.435,
                1.7902, 1.795,  1.84,   1.934,  2.07,   2.417,  2.788,  3.156,
                1.6823, 1.689,  1.733,  1.82,   1.944,  2.258,  2.596,  2.933,
                1.5929, 1.601,  1.644,  1.725,  1.838,  2.124,  2.435,  2.746,
                1.4551, 1.465,  1.504,  1.574,  1.67,   1.913,  2.181,  2.451,
                1.3551, 1.365,  1.4,    1.461,  1.544,  1.754,  1.989,  2.228,
                1.28,   1.289,  1.321,  1.374,  1.447,  1.63,   1.838,  2.053,
                1.2219, 1.231,  1.259,  1.306,  1.37,   1.532,  1.718,  1.912,
                1.1757, 1.184,  1.209,  1.251,  1.307,  1.451,  1.618,  1.795,
                1.0933, 1.1,    1.119,  1.15,   1.193,  1.304,  1.435,  1.578,
                1.0388, 1.044,  1.059,  1.083,  1.117,  1.204,  1.31,   1.428,
                0.99963, 1.004, 1.016,  1.035,  1.062,  1.133,  1.22,   1.319,
                0.96988, 0.9732, 0.983,  0.9991, 1.021,  1.079,  1.153,  1.236,
                0.92676, 0.9291, 0.936,  0.9473, 0.9628, 1.005,  1.058,  1.121,
                0.89616, 0.8979, 0.903,  0.9114, 0.923,  0.9545, 0.9955, 1.044,
                0.87272, 0.8741, 0.878,  0.8845, 0.8935, 0.9181, 0.9505, 0.9893,
                0.85379, 0.8549, 0.858,  0.8632, 0.8703, 0.8901, 0.9164, 0.9482,
                0.83795, 0.8388, 0.8414, 0.8456, 0.8515, 0.8678, 0.8895, 0.916,
                0.82435, 0.8251, 0.8273, 0.8308, 0.8356, 0.8493, 0.8676, 0.8901,
                0.80184, 0.8024, 0.8039, 0.8065, 0.8101, 0.8201, 0.8337, 0.8504,
                0.78363, 0.784,  0.7852, 0.7872, 0.7899, 0.7976, 0.8081, 0.8212,
                0.76834, 0.7687, 0.7696, 0.7712, 0.7733, 0.7794, 0.7878, 0.7983,
                0.75518, 0.7554, 0.7562, 0.7575, 0.7592, 0.7642, 0.7711, 0.7797,
                0.74364, 0.7438, 0.7445, 0.7455, 0.747,  0.7512, 0.7569, 0.7642,
                0.71982, 0.72,   0.7204, 0.7211, 0.7221, 0.725,  0.7289, 0.7339,
                0.70097, 0.7011, 0.7014, 0.7019, 0.7026, 0.7047, 0.7076, 0.7112,
                0.68545, 0.6855, 0.6858, 0.6861, 0.6867, 0.6883, 0.6905, 0.6932,
                0.67232, 0.6724, 0.6726, 0.6728, 0.6733, 0.6743, 0.6762, 0.6784,
                0.65099, 0.651,  0.6512, 0.6513, 0.6516, 0.6524, 0.6534, 0.6546,
                0.61397, 0.6141, 0.6143, 0.6145, 0.6147, 0.6148, 0.6148, 0.6147,
                0.5887, 0.5889, 0.5894, 0.59,   0.5903, 0.5901, 0.5895, 0.5885]


        #Find for each fixed tr the poly of deg 6 in dst approx omega22 values
        #store the poly coefs in m_o22poly
        m_o22poly = []
        for i in range(37):
            dstDeg = 6
            #Polynomial coefficients, highest power first 
            polycoefs = np.polyfit(dstTab,omegaTab[8*i:8*(i+1)],dstDeg)
            m_o22poly.append(polycoefs)

        #Find 3 referenced temp points around tr
        for i in range(37):
            if tr<trTab[i]:
                break
        i1 = max(i-1, 0)
        i2 = i1+3
        if (i2 > 36):
            i2 = 36
            i1 = i2 - 3
        #compute omega22 value for these 3 points
        values = []
        for j in range(i1,i2):
            if (dst == 0.0):
                values.append(omegaTab[8*j])
            else:
                poly6 = np.poly1d(m_o22poly[j])
                values.append(poly6(dst))

        #interpolate to find real tr value
        trTab_log = []
        for j in range(len(trTab)):
            trTab_log.append(np.log(trTab[j]))
        #print trTab_log[i1:i2], values
        om22_interp = self.quadInterp(np.log(tr), trTab_log[i1:i2], values)

        return om22_interp


    def _getCVdRspecies(self, t, species):

        models = species.thermo
        m1 = models[0]
        m2 = models[1]

        if m1.lowT < m2.lowT:
            lowRange = m1
            highRange = m2
        else:
            lowRange = m2
            highRange = m1

        low = lowRange.lowT
        mid = lowRange.highT
        high = highRange.highT

        if t < mid:
            parameters = lowRange.parameters
        else:
            parameters = highRange.parameters

        return ((parameters[0] - 1.0) + parameters[1] * t + parameters[2] * t * t \
                + parameters[3] * t * t * t + parameters[4] * t * t * t * t)


    def qinterp(self, x0, x, y):

        val1 = y[0] + (x0-x[0])*(y[1]-y[0]) / (x[1]-x[0])
        val2 = y[1] + (x0-x[1])*(y[2]-y[1]) / (x[2]-x[1])
        fac1 = (x0-x[0]) / (x[1]-x[0]) / 2.0
        fac2 = (x[2]-x0) / (x[2]-x[1]) / 2.0
        if (x0 >= x[1]):
           val = (val1*fac2+val2) / (1.0+fac2)
        else:
           val = (val1+val2*fac1) / (1.0+fac1)
        return val


    def quadInterp(self, x0, x, y):

        dx21 = x[1] - x[0] 
        dx32 = x[2] - x[1]
        dx31 = dx21 + dx32
        dy32 = y[2] - y[1]
        dy21 = y[1] - y[0]
        a = (dx21*dy32 - dy21*dx32)/(dx21*dx31*dx32)
        return a*(x0 - x[0])*(x0 - x[1]) + (dy21/dx21)*(x0 - x[1]) + y[1]


    ################################
    # END TRANSPORT #
    ################################


    def _emptygjs(self, mechanism):
        self._write()
        self._write(self.line(' Replace this routine with the one generated by the Gauss Jordan solver of DW'))
        self._write('AMREX_GPU_HOST_DEVICE void sgjsolve(double* A, double* x, double* b) {')
        self._indent()
        self._write('amrex::Abort("sgjsolve not implemented, choose a different solver ");')
        self._outdent()
        self._write('}')

        self._write()
        self._write(self.line(' Replace this routine with the one generated by the Gauss Jordan solver of DW'))
        self._write('AMREX_GPU_HOST_DEVICE void sgjsolve_simplified(double* A, double* x, double* b) {')
        self._indent()
        self._write('amrex::Abort("sgjsolve_simplified not implemented, choose a different solver ");')
        self._outdent()
        self._write('}')


    #Pieces for mechanism.h#
    def _print_mech_header(self, mechanism):
        self._write()
        self._write("#ifndef MECHANISM_h")
        self._write("#define MECHANISM_h")
        self._write()
        self._write("#if 0")
        self._write("/* Elements")
        nb_elem = 0
        for element in mechanism.element():
            self._write('%d  %s' % (element.id, element.symbol) )
            nb_elem += 1
        self._write('*/')
        self._write("#endif")
        self._write()
        self._write('/* Species */')
        nb_spec = 0
        for species in mechanism.species():
            if species.symbol not in self.qss_species:
                s = species.symbol.strip()
                # Ionic species
                if s[-1] == '-': s = s[:-1] + 'n'
                if s[-1] == '+': s = s[:-1] + 'p'
                # Excited species
                s = s.replace('*', 'D')
                # Remove other characters not allowed in preprocessor defines
                s = s.replace('-', '').replace('(','').replace(')','')
                self._write('#define %s_ID %d' % (s, species.id))
                nb_spec += 1
        self._write()
        self._write("#define NUM_ELEMENTS %d" % (nb_elem))
        self._write("#define NUM_SPECIES %d" % (nb_spec))
        self._write("#define NUM_REACTIONS %d" %(len(mechanism.reaction())))
        self._write()
        self._write("#define NUM_FIT 4")
        self._write("#endif")

        return
    #Pieces for mechanism.h#


    ################################
    # MISC
    ################################

    def _forwardRate(self, mechanism, reaction):

        lt = reaction.lt
        if lt:
            import pyre
            pyre.debug.Firewall.hit("Landau-Teller reactions are not supported yet")
            return self._landau(reaction)

        dim = self._phaseSpaceUnits(reaction.reactants)

        phi_f = self._phaseSpace(mechanism, reaction.reactants)
        self._write("phi_f = %s;" % phi_f)

        thirdBody = reaction.thirdBody
        if not thirdBody:
            self._write("k_f = k_f_save[%d];" % (reaction.id-1))
            self._write("q_f = phi_f * k_f;")
            return
            
        alpha = self._enhancement(mechanism, reaction)
        self._write("alpha = %s;" % alpha)

        sri = reaction.sri
        low = reaction.low
        troe = reaction.troe

        if not low:
            self._write("k_f = alpha * k_f_save[%d];" % (reaction.id-1))
            self._write("q_f = phi_f * k_f;")
            return

        self._write("k_f = k_f_save[%d];" % (reaction.id-1))

        self._write("redP = alpha / k_f * phase_units[%d] * low_A[%d] * exp(low_beta[%d] * tc[0] - activation_units[%d] * low_Ea[%d] *invT);"
                    %(reaction.id-1,reaction.id-1,reaction.id-1,reaction.id-1,reaction.id-1))
        self._write("F = redP / (1 + redP);")

        if sri:
            self._write("logPred = log10(redP);")
            self._write("X = 1.0 / (1.0 + logPred*logPred);")
            self._write("F_sri = exp(X * log(sri_a[%d] * exp(-sri_b[%d]/T)" 
                        % (reaction.id-1,reaction.id-1))
            self._write("   +  (sri_c[%d] > 1.e-100 ? exp(T/sri_c[%d]) : 0.) )" 
                        % (reaction.id-1,reaction.id-1))
            self._write("   *  (sri_len[%d] > 3 ? sri_d[%d]*exp(sri_e[%d]*tc[0]) : 1);" 
                        % (reaction.id-1,reaction.id-1,reaction.id-1))
            self._write("F *= F_sri;")

        elif troe:
            self._write("logPred = log10(redP);")

            self._write('logFcent = log10(')
            self._write('    (fabs(troe_Tsss[%d]) > 1.e-100 ? (1-troe_a[%d])*exp(-T/troe_Tsss[%d]) : 0) '
                        % (reaction.id-1,reaction.id-1,reaction.id-1))
            self._write('    + (fabs(troe_Ts[%d]) > 1.e-100 ? troe_a[%d] * exp(-T/troe_Ts[%d]) : 0) '
                        % (reaction.id-1,reaction.id-1,reaction.id-1))
            self._write('    + (troe_len[%d] == 4 ? exp(-troe_Tss[%d] * invT) : 0) );' 
                        % (reaction.id-1,reaction.id-1))
            self._write("troe_c = -.4 - .67 * logFcent;")
            self._write("troe_n = .75 - 1.27 * logFcent;")
            self._write("troe = (troe_c + logPred) / (troe_n - .14*(troe_c + logPred));")
            self._write("F_troe = pow(10, logFcent / (1.0 + troe*troe));")
            self._write("F *= F_troe;")

        self._write("k_f *= F;")
        self._write("q_f = phi_f * k_f;")
        return


    def _reverseRate(self, mechanism, reaction):
        if not reaction.reversible:
            self._write("q_r = 0.0;")
            return

        phi_r = self._phaseSpace(mechanism, reaction.products)
        self._write("phi_r = %s;" % phi_r)

        if reaction.rlt:
            import pyre
            pyre.debug.Firewall.hit("Landau-Teller reactions are not supported yet")
            return

        if reaction.rev:

            self._write("k_r = prefactor_units_rev[%d] * rev_A[%d] * exp(rev_beta[%d] * tc[0] - activation_units_rev[%d] * rev_Ea[%d] * invT);"
                        % (reaction.id-1,reaction.id-1,reaction.id-1,reaction.id-1,reaction.id-1))

            thirdBody = reaction.thirdBody
            if thirdBody:
                self._write("k_r *= alpha;")

            self._write("q_r = phi_r * k_r;")
            return
        
        self._write("Kc = Kc_save[%d];" % (reaction.id-1))

        self._write("k_r = k_f / Kc;")
        self._write("q_r = phi_r * k_r;")

        return


    def _prefactorUnits(self, code, exponent):

        if code == "mole/cm**3":
            units = mole / cm**3
        elif code == "moles":
            units = mole / cm**3
        elif code == "molecules":
            import pyre
            units = 1.0 / avogadro / cm**3
        else:
            import pyre
            pyre.debug.Firewall.hit("unknown prefactor units '%s'" % code)
            return 1

        #print "UNITS/SECOND/EXP ", units.value, second.value, exponent
        #print " units ** exponent / second (value) " , units.value ** exponent / second.value
        #print " units ** exponent / second (returned) " , units ** exponent / second
        return units ** exponent / second


    def _activationEnergyUnits(self, code):

        if code == "cal/mole":
            units = cal / mole
        elif code == "kcal/mole":
            units = kcal /mole
        elif code == "joules/mole":
            units = J / mole
        elif code == "kjoules/mole":
            units = kJ / mole
        elif code == "kelvins":
            units = Rc * kelvin
        else:
            pyre.debug.Firewall.hit("unknown activation energy units '%s'" % code)
            return 1

        return units


    def _phaseSpace(self, mechanism, reagents):

        phi = []

        for symbol, coefficient in reagents:
            if (coefficient == "1.0"):
                conc = "sc[%d]" % mechanism.species(symbol).id
            else:
                conc = "pow(sc[%d], %f)" % (mechanism.species(symbol).id, coefficient)
            phi += [conc]

        return "*".join(phi)


    def _sortedPhaseSpace(self, mechanism, reagents):

        phi = []

        for symbol, coefficient in sorted(reagents,key=lambda x:mechanism.species(x[0]).id):
            if (float(coefficient) == 1.0):
                conc = "sc[%d]" % mechanism.species(symbol).id
            else:
                conc = "pow(sc[%d], %f)" % (mechanism.species(symbol).id, float(coefficient))
            phi += [conc]

        return "*".join(phi)


    def _phaseSpaceUnits(self, reagents):

        dim = 0.0
        for symbol, coefficient in reagents:
            dim += float(coefficient)

        return dim


    def _enhancement(self, mechanism, reaction):

        thirdBody = reaction.thirdBody
        if not thirdBody:
            import pyre
            pyre.debug.Firewall.hit("_enhancement called for a reaction without a third body")
            return

        species, coefficient = thirdBody
        efficiencies = reaction.efficiencies

        if not efficiencies:
            if species == "<mixture>":
                return "mixture"
            return "sc[%d]" % mechanism.species(species).id

        alpha = ["mixture"]
        for i, eff in enumerate(efficiencies):
            symbol, efficiency = eff
            factor = "(TB[%d][%d] - 1)" % (reaction.id-1, i)
            conc = "sc[%d]" % mechanism.species(symbol).id
            alpha.append("%s*%s" % (factor, conc))

        return " + ".join(alpha).replace('+ -','- ')


    #####################
    #QSS production rate
    #####################
    def _createSSnet(self, mechanism):

        # for each reaction in the mechanism
        for r in mechanism.reaction():
            slist = []
            # get a list of species involved in the reactants and products
            for symbol, coefficient in r.reactants:
                if symbol in self.qss_list:
                    slist.append(symbol)
            for symbol, coeffecient in r.products:
                if symbol in self.qss_list:
                    slist.append(symbol)
            # if species s1 and species s2 are in the same reaction,
            # denote they are linked in the Species-Species network
            for s1 in slist:
                for s2 in slist:
                    self.QSS_SSnet[mechanism.qss_species(s1).id][mechanism.qss_species(s2).id] = 1

    def _createSRnet(self, mechanism):

        # for each reaction in the mechanism
        for i, r in enumerate(mechanism.reaction()):
            reactant_list = []
            product_list = []
            reaction_number = i
            # get a list of species involved in the reactants and products
            for symbol, coefficient in r.reactants:
                if symbol in self.qss_list:
                    reactant_list.append(symbol)
            for	symbol,	coeffecient in r.products:
                if symbol in self.qss_list:
                    product_list.append(symbol)
            # if qss species s is in reaction number i, 
            # denote they are linked in the Species-Reaction network
            # with negative if s is a reactant(consumed) and positive if s is a product(produced)
            for	s in reactant_list:
                self.QSS_SRnet[mechanism.qss_species(s).id][reaction_number] = -1
            for s in product_list:
                self.QSS_SRnet[mechanism.qss_species(s).id][reaction_number] = 1
        
    def _setQSSneeds(self, mechanism):

        self.needs = OrderedDict()
        self.needs_count = OrderedDict()

        self.needs_running = OrderedDict()
        self.needs_count_running = OrderedDict()
        
        for i in range(self.nQSSspecies):
            needs_species = []
            count = 0
            for j in self.QSS_SC_Sj[self.QSS_SC_Si == i]:
                if j != i:
                    needs_species.append(self.qss_list[j])
                    count += 1
            self.needs[self.qss_list[i]] = needs_species
            self.needs_count[self.qss_list[i]] = count

        self.needs_running = self.needs.copy()
        self.needs_count_running = self.needs_count.copy()

        print
        print "NEEDS: "
        print(self.needs)
        print(self.needs_count)

    def _setQSSisneeded(self, mechanism):

        self.is_needed = OrderedDict()
        self.is_needed_count = OrderedDict()

        self.is_needed_running = OrderedDict()
        self.is_needed_count_running = OrderedDict()

        for i in range(self.nQSSspecies):
            is_needed_species = []
            count = 0
            for j in self.QSS_SC_Si[self.QSS_SC_Sj == i]:
                if j != i:
                    is_needed_species.append(self.qss_list[j])
                    count += 1
            self.is_needed[self.qss_list[i]] = is_needed_species
            self.is_needed_count[self.qss_list[i]] = count

        self.is_needed_running = self.is_needed.copy()
        self.is_needed_count_running = self.is_needed_count.copy()

        print
        print "IS NEEDED: "
        print(self.is_needed)
        print(self.is_needed_count)

    # get two-way dependencies accounted for: (s1 needs s2) and (s2 needs s1) = group
    def _getQSSgroups(self, mechanism):
    
        self.group = OrderedDict()
        already_accounted_for = []
        group_count = 0
        # print(self.needs)
        # print(self.is_needed)
        # print

        # for each species spec that has needs
        for spec in self.needs.keys():
            print("dealing with species: ", spec)
            print
            # for each of the needs of that species spec, we could have a potential group
            for needs in self.needs[spec]:
                print(needs)
                print(self.is_needed[spec])
                print
                potential_group = []
                potential_group.append(spec)
                # check that this group was not already found through a search with the other species involved
                if (needs,spec) not in already_accounted_for:
                    # if species spec is also needed by the current needs, then we have a group
                    if any(species == needs for species in self.is_needed[spec]):
                        print("here!")
                        print("species "+spec+" and "+needs+" depend on eachother")
                        potential_group.append(needs)
                        self.group['group_'+str(group_count)] = potential_group

                        # Add this group to a list so that it doesn't get counted again in the event that the needs species is also a spec in the needs key list
                        already_accounted_for.append((spec,needs))
                        group_count += 1
                    print(potential_group)
                    print "\n\n"
        print(self.group)

        self._updateGroupNeeds(mechanism)
        self._updateGroupDependencies(mechanism)

    # get intergroup dependencies accounted for: (this needs that) and (that needs this) = supergroup, where this and that can be individual species or groups
    def _getQSSsupergroups(self, mechanism):

        self.super_group = OrderedDict()
        already_accounted_for = []
        super_group_count = 0

        for member in self.needs_running.keys():
            for needs in self.needs_running[member]:
                potential_super_group = []
                potential_super_group.append(member)

                if(needs,member) not in already_accounted_for:

                    if any(components == needs for components in self.is_needed_running[member]):
                        potential_super_group.append(needs)
                        self.super_group['super_group_'+str(super_group_count)] = potential_super_group

                        alread_accounted_for.append((member,needs))
                        supergroup_count += 1

        print
        print("The super groups are: ", self.super_group)

        self._updateSupergroupNeeds(mechanism)
        self._updateSupergroupDependencies(mechanism)

    # Sort order that QSS species need to be computed based on dependencies
    def _sortQSScomputation(self, mechanism):

        self.decouple_index = OrderedDict()
        self.decouple_count = 0

        # look at how many dependencies each component has
        needs_count_regress = self.needs_count_running.copy()
        print
        print(needs_count_regress)

        # There should always be a component present that loses all dependencies as you update the computation
        while 0 in needs_count_regress.values():
            needs_count_base = needs_count_regress.copy()

            # for each component (species, group, sup group, etc.) that needs things... 
            for member in needs_count_base:

                # if that component doesn't need anything
                if needs_count_base[member] == 0:

                    # solve that component now
                    self.decouple_index[member] = self.decouple_count
                    # then delete it out of the updating needs list
                    del needs_count_regress[member]

                    # for anything needed by that component
                    for needed in self.is_needed_running[member]:
                        # decrease that thing's dependency since it has now been taken care of
                        needs_count_regress[needed] -= 1
                        
                    self.decouple_count += 1
                    
        # If your decouple count doesn't match the number of components with needs, then the system is more complicated than what these functions can handle currently
        if len(self.decouple_index) != len(self.needs_running):
            print("WARNING: Some components may not have been taken into account")

        print
        print "ORDER OF EXECUTION FOR QSS CONCENTRATION CALCULATIONS: "
        print(self.decouple_index)

    # Update group member needs with group names:
    # group member needs species -> group needs species
    # group member is needed by species -> group is needed by species
    def _updateGroupNeeds(self, mechanism):

        for group_key in self.group.keys():
            
            update_needs = []
            update_is_needed = []
            update_needs_count = 0
            update_needed_count = 0

            group_needs = {}
            group_needs_count = {}
            group_is_needed = {}
            group_is_needed_count = {}
            
            other_groups = self.group.keys()
            other_groups.remove(group_key)
            
            # print("other groups are: ", other_groups)
            # print("we are in group: ", group_key)
            # print(self.group[group_key])

            # for each species in the current group
            for spec in self.group[group_key]:
                print("for group member: ",spec)
                # look at any additional needs that are not already accounted for with the group
                for need in list(set(self.needs_running[spec]) - set(self.group[group_key])):
                    print("need is ", need)
                    not_in_group = True
                    # check the other groups to see if the need can be found in one of them
                    for other_group in other_groups:
                        # if the other group is not already accounted for and it contains the need we're looking for, update the group needs with that group that contains the need
                        if other_group not in update_needs and any(member == need for member in self.group[other_group]):
                            print("this is in a different group")
                            not_in_group = False
                            update_needs.append(other_group)
                            update_needs_count += 1
                    # alternatively, if this is just a solo need that's not in another group, update the group needs with just that need. 
                    if not_in_group and need not in update_needs:
                        print("Need ", need, " was not in a group")
                        update_needs.append(need)
                        update_needs_count += 1
                # look at any additional species (outside of the group) that depend on the current group member
                for needed in list(set(self.is_needed_running[spec]) - set(self.group[group_key])):
                    not_in_group = True
                    # for the other groups
                    for other_group in other_groups:
                        # if the other group hasn't alredy been accounted for and the species is in that group, then that other group depends on a species in the current group
                        if other_group not in update_is_needed and any(member == needed for member in self.group[other_group]):
                            not_in_group = False
                            update_is_needed.append(other_group)
                            update_needed_count += 1
                    # if the species is not in another group, then that lone species just depends on the current group. 
                    if not_in_group and needed not in update_is_needed:
                        update_is_needed.append(needed)
                        update_needed_count += 1

                del self.needs_running[spec]
                del self.needs_count_running[spec]
                del self.is_needed_running[spec]

            group_needs[group_key] = update_needs
            group_needs_count[group_key] = update_needs_count
            group_is_needed[group_key] = update_is_needed
            group_is_needed_count[group_key] = update_needed_count

            self.needs_running.update(group_needs)
            self.needs_count_running.update(group_needs_count)
            self.is_needed_running.update(group_is_needed)
            self.is_needed_count_running.update(group_is_needed_count)

        print
        print(self.needs_running)
        print(self.needs_count_running)
        print(self.is_needed_running)
        print(self.is_needed_count_running)

        
    # Update solo species dependendent on group members with group names:
    # species needs member -> species needs group
    # species is needed by group member -> species is needed by group
    def _updateGroupDependencies(self, mechanism):

        solo_needs = self.needs_running.copy()
        solo_needs_count = self.needs_count_running.copy()
        solo_is_needed = self.is_needed_running.copy()
        solo_is_needed_count = self.is_needed_count_running.copy()

        # remove the groups because we're just dealing with things that aren't in groups now
        for group in self.group.keys():
            del solo_needs[group]
            del solo_needs_count[group]
            del solo_is_needed[group]
            del solo_is_needed_count[group]

        print
        print(solo_needs)
        print(solo_is_needed)
        
        for solo in solo_needs.keys():
            
            update_needs = []
            update_is_needed = []
            update_needs_count = 0
            update_needed_count = 0

            for need in solo_needs[solo]:
                print(need)
                not_in_group = True
                for group in self.group.keys():
                    if group not in update_needs and any(member == need for member in self.group[group]):
                        print("Species ", need, " is in group: ", group)
                        not_in_group = False

                        update_needs.append(group)
                        update_needs_count += 1
                if not_in_group and need not in update_needs:

                    update_needs.append(need)
                    update_needs_count += 1

            for needed in solo_is_needed[solo]:
                not_in_group = True
                for group in self.group.keys():
                    if group not in update_is_needed and any(member == needed for member in self.group[group]):
                        not_in_group = False

                        update_is_needed.append(group)
                        update_needed_count +=1

                if not_in_group and needed not in update_is_needed:

                    update_is_needed.append(needed)
                    update_needed_count += 1

            solo_needs[solo] = update_needs
            solo_needs_count[solo] = update_needs_count
            solo_is_needed[solo] = update_is_needed
            solo_is_needed_count[solo] = update_needed_count

        self.needs_running.update(solo_needs)
        self.needs_count_running.update(solo_needs_count)
        self.is_needed_running.update(solo_is_needed)
        self.is_needed_count_running.update(solo_is_needed_count)

        print
        print(self.needs_running)
        print(self.needs_count_running)
        print(self.is_needed_running)
        print(self.is_needed_count_running)


    ###############################################################
    #
    # Updates for Super Groups are essentially the same as those
    # for Groups. Have not actually been tested yet because my baby
    # example isn't that complicated
    #
    ###############################################################
        
                        
    def _updateSupergroupNeeds(self, mechanism):
        
        for super_group_key in self.super_group.keys():
            
            update_needs = []
            update_is_needed = []
            update_needs_count = 0
            update_needed_count = 0

            super_group_needs = {}
            super_group_needs_count = {}
            super_group_is_needed = {}
            super_group_is_needed_count = {}
            
            other_super_groups = self.super_group.keys()
            other_super_groups.remove(super_group_key)
            
            # print("other super groups are: ", other_super_groups)
            # print("we are in supergroup: ", super_group_key)
            # print(self.supergroup[super_group_key])

            # for each species in the current group
            for spec in self.super_group[super_group_key]:
                print("for super group member: ",spec)
                # look at any additional needs that are not already accounted for with the group
                for need in list(set(self.needs_running[spec]) - set(self.super_group[super_group_key])):
                    print("need is ", need)
                    not_in_super_group = True
                    # check the other groups to see if the need can be found in one of them
                    for other_super_group in other_super_groups:
                        # if the other group is not already accounted for and it contains the need we're looking for, update the group needs with that group that contains the need
                        if other_super_group not in update_needs and any(member == need for member in self.super_group[other_super_group]):
                            print("this is in a different super group")
                            not_in_super_group = False
                            update_needs.append(other_super_group)
                            update_needs_count += 1
                    # alternatively, if this is just a solo need that's not in another group, update the group needs with just that need. 
                    if not_in_super_group and need not in update_needs:
                        print("Need ", need, " was not in a super group")
                        update_needs.append(need)
                        update_needs_count += 1
                # look at any additional species (outside of the group) that depend on the current group member
                for needed in list(set(self.is_needed_running[spec]) - set(self.super_group[super_group_key])):
                    not_in_super_group = True
                    # for the other groups
                    for other_super_group in other_super_groups:
                        # if the other group hasn't alredy been accounted for and the species is in that group, then that other group depends on a species in the current group
                        if other_super_group not in update_is_needed and any(member == needed for member in self.super_group[other_super_group]):
                            not_in_super_group = False
                            update_is_needed.append(other_super_group)
                            update_needed_count += 1
                    # if the species is not in another group, then that lone species just depends on the current group. 
                    if not_in_super_group and needed not in update_is_needed:
                        update_is_needed.append(needed)
                        update_needed_count += 1

                del self.needs_running[spec]
                del self.needs_count_running[spec]
                del self.is_needed_running[spec]

            super_group_needs[super_group_key] = update_needs
            super_group_needs_count[super_group_key] = update_needs_count
            super_group_is_needed[super_group_key] = update_is_needed
            super_group_is_needed_count[super_group_key] = update_needed_count

            self.needs_running.update(super_group_needs)
            self.needs_count_running.update(super_group_needs_count)
            self.is_needed_running.update(super_group_is_needed)
            self.is_needed_count_running.update(super_group_is_needed_count)

        print
        print(self.needs_running)
        print(self.needs_count_running)
        print(self.is_needed_running)
        print(self.is_needed_count_running)

    # Update solo species dependendent on group members with group names:
    # species needs member -> species needs group
    # species is needed by group member -> species is needed by group
    def _updateSupergroupDependencies(self, mechanism):

        super_solo_needs = self.needs_running.copy()
        super_solo_needs_count = self.needs_count_running.copy()
        super_solo_is_needed = self.is_needed_running.copy()
        super_solo_is_needed_count = self.is_needed_count_running.copy()

        # remove the groups because we're just dealing with things that aren't in groups now
        for super_group in self.super_group.keys():
            del super_solo_needs[super_group]
            del super_solo_needs_count[super_group]
            del super_solo_is_needed[super_group]
            del super_solo_is_needed_count[super_group]

        print
        print(super_solo_needs)
        print(super_solo_is_needed)
        
        for solo in super_solo_needs.keys():
            
            update_needs = []
            update_is_needed = []
            update_needs_count = 0
            update_needed_count = 0

            for need in super_solo_needs[solo]:
                print(need)
                not_in_super_group = True
                for super_group in self.super_group.keys():
                    if super_group not in update_needs and any(member == need for member in self.super_group[super_group]):
                        print("Species ", need, " is in super group: ", super_group)
                        not_in_super_group = False

                        update_needs.append(super_group)
                        update_needs_count += 1
                if not_in_super_group and need not in update_needs:

                    update_needs.append(need)
                    update_needs_count += 1

            for needed in super_solo_is_needed[solo]:
                not_in_super_group = True
                for super_group in self.super_group.keys():
                    if super_group not in update_is_needed and any(member == needed for member in self.super_group[super_group]):
                        not_in_super_group = False

                        update_is_needed.append(super_group)
                        update_needed_count +=1

                if not_in_super_group and needed not in update_is_needed:

                    update_is_needed.append(needed)
                    update_needed_count += 1

            super_solo_needs[solo] = update_needs
            super_solo_needs_count[solo] = update_needs_count
            super_solo_is_needed[solo] = update_is_needed
            super_solo_is_needed_count[solo] = update_needed_count

        self.needs_running.update(super_solo_needs)
        self.needs_count_running.update(super_solo_needs_count)
        self.is_needed_running.update(super_solo_is_needed)
        self.is_needed_count_running.update(super_solo_is_needed_count)

        print
        print(self.needs_running)
        print(self.needs_count_running)
        print(self.is_needed_running)
        print(self.is_needed_count_running)
                        
    ###################################################################
    #
    ###################################################################
        
    # Check that the QSS given are actually valid options
    # Exit if species is not valid
    def _QSSvalidation(self, mechanism):

        species = []
        for s in mechanism.species():
            species.append(s.symbol)

        # NOT DOABLE ANYMORE. QSS ARE READ IN SEPARATELY FROM TRANSPORTED SPECIES
        # # Check that QSS species are all species used in the given mechanism
        # for s in self.qss_species:
        #     if s not in species:
        #         text = 'species '+s+' is not in the mechanism'
        #         sys.exit(text)

        # sets up QSS subnetwork to be used for coupling too
        self._getQSSnetworks(mechanism)

        # Check that QSS species are consumed/produced at least once to ensure theoretically valid QSS option
        # (There is more to it than that, but this is a quick catch based on that aspect)
        for i, symbol in enumerate(self.qss_list):
            consumed = 0
            produced = 0
            for j in self.QSS_SR_Rj[self.QSS_SR_Si == i]:
                reaction = mechanism.reaction(id=j)

                if any(reactant == symbol for reactant,_ in list(set(reaction.reactants))):
                    consumed += 1
                    if reaction.reversible:
                        produced += 1
                if any(product == symbol for product,_ in list(set(reaction.products))):
                    produced += 1
                    if reaction.reversible:
                        consumed += 1

            if consumed == 0 or produced == 0:
                text = 'Uh Oh! QSS species '+symbol+' does not have a balanced consumption/production relationship in mechanism => bad QSS choice'
                sys.exit(text)


    # Get networks showing which QSS species are involved with one another and which reactions each QSS species is involved in
    def _getQSSnetworks(self, mechanism):
        # Create QSS networks for mechanism
        self._createSSnet(mechanism)
        self._createSRnet(mechanism)

        # Get non-zero indices to be used for coupling
        self.QSS_SS_Si, self.QSS_SS_Sj = np.nonzero(self.QSS_SSnet)
        self.QSS_SR_Si, self.QSS_SR_Rj = np.nonzero(self.QSS_SRnet)

        print("\n\n SS network for QSS: ")
        print(self.QSS_SSnet)
        print("SR network for QSS: ")
        print(self.QSS_SRnet)

        # print(self.QSS_SS_Si)
        # print(self.QSS_SS_Sj)
        # print(self.QSS_SR_Si)
        # print(self.QSS_SR_Rj)

        
    # Determine from QSS_SSnet which QSS species depend on each other specifically
    def _QSSCoupling(self, mechanism):

        self.QSS_SCnet = self.QSS_SSnet

        for i in range(self.nQSSspecies):
            for j in self.QSS_SS_Sj[self.QSS_SS_Si == i]:
                if j != i:
                    # print("looking for coupling with species ", j, "(", self.qss_list[j], ")")
                    count = 0
                    for r in self.QSS_SR_Rj[self.QSS_SR_Si == j]:
                        reaction = mechanism.reaction(id=r)

                        #Check if QSS species i and QSS species j are both reactants in the reaction containing QSS species j
                        if any(reactant == self.qss_list[j] for reactant,_ in  list(set(reaction.reactants))):
                            if any(reactant == self.qss_list[i] for reactant,_ in list(set(reaction.reactants))):
                                sys.exit('Quadratic coupling between '+self.qss_list[j]+' and '+self.qss_list[i]+' not allowed !!!')
                            # if QSS specices j is a reactant and QSS species i is a product, then i depends on j
                            elif any(product == self.qss_list[i] for product,_ in list(set(reaction.products))):
                                count += 1
                        # if QSS species j is not a reactant, then it must be a product. Check if QSS species i is also a product
                        elif reaction.reversible:
                            if any(product == self.qss_list[i] for product,_ in list(set(reaction.products))):
                                sys.exit('Quadratic coupling between '+self.qss_list[j]+' and '+self.qss_list[i]+' not allowed !!!')
                            # if QSS species i is a reactant in this reversible reaction, then i depends on j 
                            elif any(reactant == self.qss_list[i] for reactant,_ in list(set(reaction.reactants))):
                                count += 1
                                
                    if count == 0:
                        self.QSS_SCnet[i,j] = 0
                        
        self.QSS_SC_Si, self.QSS_SC_Sj = np.nonzero(self.QSS_SCnet)
        print(self.QSS_SCnet)
        print

    # Components needed to set up QSS algebraic expressions from AX = B, where A contains coeffiencts from qf's and qr's, X contains QSS species concentrations, and B contains:
    # RHS vector (non-QSS and QSS qf's and qr's), coefficient of species (diagonal elements of A), coefficient of group mates (coupled off-diagonal elements of A)
    def _sortQSSsolution_elements(self, mechanism):

        self.QSS_rhs = OrderedDict()
        self.QSS_coeff = OrderedDict()
        
        for i in range(self.nQSSspecies):
            coupled = []
            rhs_hold = []
            coeff_hold = []
            for r in self.QSS_SR_Rj[self.QSS_SR_Si == i]:

                reaction = mechanism.reaction(id=r)
                direction = self.QSS_SRnet[i][r]
                
                # Check if reaction contains other QSS species
                coupled = [species for species in list(set(self.QSS_SR_Si[self.QSS_SR_Rj == r])) if species != i]

                print "COUPLED IS: "
                print coupled
                
                if not coupled:
                    print("reaction ", r, " only contains QSS species ", self.qss_list[i])
                    print
                    # if QSS species is a reactant
                    if direction == -1:
                        print("for species ", self.qss_list[i], " in reaction ", r, " is a reactant")
                        if reaction.reversible:
                            print("MOVE THIS SPECIES TO RHS")
                            print
                            print
                            rhs_hold.append('-qb['+str(r)+']')
                        else:
                            print("not reversible => qr = 0")
                            print
                            print
                        coeff_hold.append('-qf['+str(r)+']')
                    # if QSS species is a product
                    elif direction == 1:
                        if reaction.reversible:
                            coeff_hold.append('-qf['+str(r)+']')
                        print("for species ", self.qss_list[i], " in reaction ", r, " is a product")
                        print("MOVE THIS SPECIES TO RHS")
                        rhs_hold.append('-qf['+str(r)+']')
                else:
                    # print("reaction ", r, " only contains QSS species ", self.qss_list[i])
                    print
                    # if QSS species is a reactant
                    if direction == -1:
                        print("for species ", self.qss_list[i], " in reaction ", r, " is a reactant")
                        if reaction.reversible:
                            print("MOVE THIS SPECIES TO RHS")
                            print
                            print
                            rhs_hold.append('-qb['+str(r)+']')
                        else:
                            print("not reversible => qr = 0")
                            print
                            print
                        coeff_hold.append('-qf['+str(r)+']')
                    # if QSS species is a product
                    elif direction == 1:
                        if reaction.reversible:
                            coeff_hold.append('-qf['+str(r)+']')
                        print("for species ", self.qss_list[i], " in reaction ", r, " is a product")
                        print("MOVE THIS SPECIES TO RHS")
                        rhs_hold.append('-qf['+str(r)+']')
                                                
            self.QSS_rhs[self.qss_list[i]] = " ".join(rhs_hold)

        print(self.QSS_rhs)
        
                        
                
        
    # Testing all of my garbage for now
    # Order of operation as of right now: run setQSSspecies, QSSvalidation, then QSSCoupling
    def _test(self, mechanism):
        # Testing how this works in general
        # self._write(self.line('Does this write to the file?'))
        # print("\n\nDoes this print to the terminal when run?\n\n")

        # Testing setQSSspecies to initialize QSS information
        # QSS = ['O2', 'CH', 'CH2', 'C', 'H2O', 'H2O2', 'CO', 'HO2']
        # QSS = ['C','HCCO']

        # self._setQSSspecies(mechanism, QSS)
        self._QSSvalidation(mechanism)
        # print("Global QSS info is: ", self.QSS_species)
        # print("QSS_SSnet is starting with: ", self.QSS_SSnet)
        # print("QSS_SRnet is starting with: ", self.QSS_SRnet)

        # # Test helper functions for getting QSS subnetwork
        # self._createSSnet(mechanism)
        # print("\nEvaluated SSnet is: \n")
        # print(self.QSS_SSnet)
        # print()
        # self._createSRnet(mechanism)
        # print("\nEvaluated SRnet is: \n")
        # print(self.QSS_SRnet)
        # print()

        # Testing creation of QSS Subnetworks for mechanism
        # self._getQSSsubnetworks(mechanism)

        # Testing QSS Coupling
        self._QSSCoupling(mechanism)

        self._setQSSneeds(mechanism)
        self._setQSSisneeded(mechanism)

        self._getQSSgroups(mechanism)
        self._getQSSsupergroups(mechanism)

        self._sortQSScomputation(mechanism)

        self._sortQSSsolution_elements(mechanism)

    def _test2(self, mechanism):
        for species in self.species:
            print (species.symbol, " ",species.id, " ", species.weight)
        for qss in self.qss_species:
            print(qss.symbol, " ", qss.id, " ", qss.weight)

    # Not actually sorted right now, just regular PhaseSpace with QSS distinction
    def _QSSsortedPhaseSpace(self, mechanism, reagents):

        phi = []

        for symbol, coefficient in reagents:
            if symbol in self.qss_list:
                if (float(coefficient) == 1.0):
                    conc = "qss_sc[%d]" % mechanism.qss_species(symbol).id
                else:
                    conc = "pow(qss_sc[%d], %f)" % (mechanism.qss_species(symbol).id, float(coefficient))
                phi += [conc]
            else:
                if (float(coefficient) == 1.0):
                    conc = "sc[%d]" % mechanism.species(symbol).id
                else:
                    conc = "pow(sc[%d], %f)" % (mechanism.species(symbol).id, float(coefficient))
                phi += [conc]
        return "*".join(phi)


    def _QSScomponentFunctions(self, mechanism):

        nSpecies = len(mechanism.species())
        nReactions = len(mechanism.reaction())

        itroe      = self.reactionIndex[0:2]
        isri       = self.reactionIndex[1:3]
        ilindemann = self.reactionIndex[2:4]
        i3body     = self.reactionIndex[3:5] 
        isimple    = self.reactionIndex[4:6]
        ispecial   = self.reactionIndex[5:7]

        if len(self.reactionIndex) != 7:
            print '\n\nCheck this!!!\n'
            sys.exit(1)
        
        ntroe      = itroe[1]      - itroe[0]
        nsri       = isri[1]       - isri[0]
        nlindemann = ilindemann[1] - ilindemann[0]
        n3body     = i3body[1]     - i3body[0]
        nsimple    = isimple[1]    - isimple[0]
        nspecial   = ispecial[1]   - ispecial[0]
        
        # qss_coefficients
        self._write()
        self._write('void comp_qss_qfqr_coeff(double *  qf_co, double *  qr_co, double *  sc, double * qss_sc, double *  tc, double invT)')
        self._write('{')
        self._indent()

        nclassd = nReactions - nspecial
        nCorr   = n3body + ntroe + nsri + nlindemann

        for i in range(nclassd):
            self._write()
            reaction = mechanism.reaction(id=i)
            self._write(self.line('reaction %d: %s' % (reaction.id, reaction.equation())))
            if (len(reaction.ford) > 0):
                self._write("qf_co[%d] = %s;" % (i, self._QSSsortedPhaseSpace(mechanism, reaction.ford)))
            else:
                self._write("qf_co[%d] = %s;" % (i, self._QSSsortedPhaseSpace(mechanism, reaction.reactants)))
            if reaction.reversible:
                self._write("qr_co[%d] = %s;" % (i, self._QSSsortedPhaseSpace(mechanism, reaction.products)))
            else:
                self._write("qr_co[%d] = 0.0;" % (i))

        self._write()
        self._write('double T = tc[1];')
        self._write()
        self._write(self.line('compute the mixture concentration'))
        self._write('double mixture = 0.0;')
        self._write('for (int i = 0; i < %d; ++i) {' % nSpecies)
        self._indent()
        self._write('mixture += sc[i];')
        self._outdent()
        self._write('}')

        self._write()
        self._write("double Corr[%d];" % nclassd)
        self._write('for (int i = 0; i < %d; ++i) {' % nclassd)
        self._indent()
        self._write('Corr[i] = 1.0;')
        self._outdent()
        self._write('}')

        if ntroe > 0:
            self._write()
            self._write(self.line(" troe"))
            self._write("{")
            self._indent()
            self._write("double alpha[%d];" % ntroe)
            alpha_d = {}
            for i in range(itroe[0],itroe[1]):
                ii = i - itroe[0]
                reaction = mechanism.reaction(id=i)
                if reaction.thirdBody:
                    alpha = self._enhancement(mechanism, reaction)
                    if alpha in alpha_d:
                        self._write("alpha[%d] = %s;" %(ii,alpha_d[alpha]))
                    else:
                        self._write("alpha[%d] = %s;" %(ii,alpha))
                        alpha_d[alpha] = "alpha[%d]" % ii

            if ntroe >= 4:
                self._outdent()
                self._outdent()
                self._write('#ifdef __INTEL_COMPILER')
                self._indent()
                self._indent()
                self._write(' #pragma simd')
                self._outdent()
                self._outdent()
                self._write('#endif')
                self._indent()
                self._indent()
            self._write("for (int i=%d; i<%d; i++)" %(itroe[0],itroe[1]))
            self._write("{")
            self._indent()
            self._write("double redP, F, logPred, logFcent, troe_c, troe_n, troe, F_troe;")
            self._write("redP = alpha[i-%d] / k_f_save[i] * phase_units[i] * low_A[i] * exp(low_beta[i] * tc[0] - activation_units[i] * low_Ea[i] *invT);" % itroe[0])
            self._write("F = redP / (1.0 + redP);")
            self._write("logPred = log10(redP);")
            self._write('logFcent = log10(')
            self._write('    (fabs(troe_Tsss[i]) > 1.e-100 ? (1.-troe_a[i])*exp(-T/troe_Tsss[i]) : 0.) ')
            self._write('    + (fabs(troe_Ts[i]) > 1.e-100 ? troe_a[i] * exp(-T/troe_Ts[i]) : 0.) ')
            self._write('    + (troe_len[i] == 4 ? exp(-troe_Tss[i] * invT) : 0.) );')
            self._write("troe_c = -.4 - .67 * logFcent;")
            self._write("troe_n = .75 - 1.27 * logFcent;")
            self._write("troe = (troe_c + logPred) / (troe_n - .14*(troe_c + logPred));")
            self._write("F_troe = pow(10., logFcent / (1.0 + troe*troe));")
            self._write("Corr[i] = F * F_troe;")
            self._outdent()
            self._write('}')

            self._outdent()
            self._write("}")

        if nsri > 0:
            self._write()
            self._write(self.line(" SRI"))
            self._write("{")
            self._indent()
            self._write("double alpha[%d];" % nsri)
            self._write("double redP, F, X, F_sri;")
            alpha_d = {}
            for i in range(isri[0],isri[1]):
                ii = i - isri[0]
                reaction = mechanism.reaction(id=i)
                if reaction.thirdBody:
                    alpha = self._enhancement(mechanism, reaction)
                    if alpha in alpha_d:
                        self._write("alpha[%d] = %s;" %(ii,alpha_d[alpha]))
                    else:
                        self._write("alpha[%d] = %s;" %(ii,alpha))
                        alpha_d[alpha] = "alpha[%d]" % ii

            if nsri >= 4:
                self._outdent()
                self._outdent()
                self._write('#ifdef __INTEL_COMPILER')
                self._indent()
                self._indent()
                self._write(' #pragma simd')
                self._outdent()
                self._outdent()
                self._write('#endif')
                self._indent()
                self._indent()
            self._write("for (int i=%d; i<%d; i++)" %(isri[0],isri[1]))
            self._write("{")
            self._indent()
            self._write("redP = alpha[i-%d] / k_f_save[i] * phase_units[i] * low_A[i] * exp(low_beta[i] * tc[0] - activation_units[i] * low_Ea[i] *invT);" % itroe[0])
            self._write("F = redP / (1.0 + redP);")
            self._write("logPred = log10(redP);")
            self._write("X = 1.0 / (1.0 + logPred*logPred);")
            self._write("F_sri = exp(X * log(sri_a[i] * exp(-sri_b[i]*invT)")
            self._write("   +  (sri_c[i] > 1.e-100 ? exp(T/sri_c[i]) : 0.0) )")
            self._write("   *  (sri_len[i] > 3 ? sri_d[i]*exp(sri_e[i]*tc[0]) : 1.0);")
            self._write("Corr[i] = F * F_sri;")
            self._outdent()
            self._write('}')

            self._outdent()
            self._write("}")

        if nlindemann > 0:
            self._write()
            self._write(self.line(" Lindemann"))
            self._write("{")
            self._indent()
            if nlindemann > 1:
                self._write("double alpha[%d];" % nlindemann)
            else:
                self._write("double alpha;")

            for i in range(ilindemann[0],ilindemann[1]):
                ii = i - ilindemann[0]
                reaction = mechanism.reaction(id=i)
                if reaction.thirdBody:
                    alpha = self._enhancement(mechanism, reaction)
                    if nlindemann > 1:
                        self._write("alpha[%d] = %s;" %(ii,alpha))
                    else:
                        self._write("alpha = %s;" %(alpha))

            if nlindemann == 1:
                self._write("double redP = alpha / k_f_save[%d] * phase_units[%d] * low_A[%d] * exp(low_beta[%d] * tc[0] - activation_units[%d] * low_Ea[%d] * invT);" 
                            % (ilindemann[0],ilindemann[0],ilindemann[0],ilindemann[0],ilindemann[0],ilindemann[0]))
                self._write("Corr[%d] = redP / (1. + redP);" % ilindemann[0])
            else:
                if nlindemann >= 4:
                    self._outdent()
                    self._write('#ifdef __INTEL_COMPILER')
                    self._indent()
                    self._write(' #pragma simd')
                    self._outdent()
                    self._write('#endif')
                    self._indent()
                self._write("for (int i=%d; i<%d; i++)" % (ilindemann[0], ilindemann[1]))
                self._write("{")
                self._indent()
                self._write("double redP = alpha[i-%d] / k_f_save[i] * phase_units[i] * low_A[i] * exp(low_beta[i] * tc[0] - activation_units[i] * low_Ea[i] * invT);"
                            % ilindemann[0])
                self._write("Corr[i] = redP / (1. + redP);")
                self._outdent()
                self._write('}')

            self._outdent()
            self._write("}")

        if n3body > 0:
            self._write()
            self._write(self.line(" simple three-body correction"))
            self._write("{")
            self._indent()
            self._write("double alpha;")
            alpha_save = ""
            for i in range(i3body[0],i3body[1]):
                reaction = mechanism.reaction(id=i)
                if reaction.thirdBody:
                    alpha = self._enhancement(mechanism, reaction)
                    if alpha != alpha_save:
                        alpha_save = alpha
                        self._write("alpha = %s;" % alpha)
                    self._write("Corr[%d] = alpha;" % i)
            self._outdent()
            self._write("}")

        self._write()
        self._write("for (int i=0; i<%d; i++)" % nclassd)
        self._write("{")
        self._indent()
        self._write("qf_co[i] *= Corr[i] * k_f_save[i];")
        self._write("qr_co[i] *= Corr[i] * k_f_save[i] / Kc_save[i];")
        self._outdent()
        self._write("}")
        
        if nspecial > 0:

            print "\n\n ***** WARNING: %d unclassified reactions\n" % nspecial

            self._write()
            self._write(self.line('unclassified reactions'))
            self._write('{')
            self._indent()

            self._write(self.line("reactions: %d to %d" % (ispecial[0]+1,ispecial[1])))

            #self._write('double Kc;                      ' + self.line('equilibrium constant'))
            self._write('double k_f;                     ' + self.line('forward reaction rate'))
            self._write('double k_r;                     ' + self.line('reverse reaction rate'))
            self._write('double q_f;                     ' + self.line('forward progress rate'))
            self._write('double q_r;                     ' + self.line('reverse progress rate'))
            self._write('double phi_f;                   '
                        + self.line('forward phase space factor'))
            self._write('double phi_r;                   ' + self.line('reverse phase space factor'))
            self._write('double alpha;                   ' + self.line('enhancement'))

            for i in range(ispecial[0],ispecial[1]):
                self._write()
                reaction = mechanism.reaction(id=i)
                self._write(self.line('reaction %d: %s' % (reaction.id, reaction.equation())))

                # compute the rates
                self._forwardRate(mechanism, reaction)
                self._reverseRate(mechanism, reaction)

                # store the progress rate
                self._write("qf[%d] = q_f;" % i)
                self._write("qr[%d] = q_r;" % i)

            self._outdent()
            self._write('}')

        self._write()
        self._write('return;')
        self._outdent()
        self._write('}')

        return


    ####################
    #unused
    ####################

    def _initializeRateCalculationFR(self, mechanism):

        nSpecies = len(mechanism.species())
        nReactions = len(mechanism.reaction())

        # declarations
        self._write()
        self._write('int id; ' + self.line('loop counter'))

        self._write('double mixture;                 '
                    + self.line('mixture concentration'))
        self._write('double g_RT[%d];                ' % nSpecies
                    + self.line('Gibbs free energy'))

        self._write('double Kc;                      ' + self.line('equilibrium constant'))
        self._write('double k_f;                     ' + self.line('forward reaction rate'))
        self._write('double k_r;                     ' + self.line('reverse reaction rate'))
        self._write('double phi_f;                   '
                    + self.line('forward phase space factor'))
        self._write('double phi_r;                   '
                    + self.line('reverse phase space factor'))
        self._write('double alpha;                   ' + self.line('enhancement'))


        self._write('double redP;                    ' + self.line('reduced pressure'))
        self._write('double logPred;                 ' + self.line('log of above'))
        self._write('double F;                       '
                    + self.line('fallof rate enhancement'))
        self._write()
        self._write('double F_troe;                  ' + self.line('TROE intermediate'))
        self._write('double logFcent;                ' + self.line('TROE intermediate'))
        self._write('double troe;                    ' + self.line('TROE intermediate'))
        self._write('double troe_c;                  ' + self.line('TROE intermediate'))
        self._write('double troe_n;                  ' + self.line('TROE intermediate'))
        self._write()

        self._write(
            'double tc[] = { log(T), T, T*T, T*T*T, T*T*T*T }; '
            + self.line('temperature cache'))

        self._write()
        self._write('double invT = 1.0 / tc[1];')

        # compute the reference concentration
        self._write()
        self._write(self.line('reference concentration: P_atm / (RT) in inverse mol/m^3'))
        self._write('double refC = %g / %g / T;' % (atm.value, R.value))

        # compute the mixture concentration
        self._write()
        self._write(self.line('compute the mixture concentration'))
        self._write('mixture = 0.0;')
        self._write('for (id = 0; id < %d; ++id) {' % nSpecies)
        self._indent()
        self._write('mixture += sc[id];')
        self._outdent()
        self._write('}')

        # compute the Gibbs free energies
        self._write()
        self._write(self.line('compute the Gibbs free energy'))
        self._write('gibbs(g_RT, tc);')
        return


    def _sortedKc(self, mechanism, reaction):
        conv = self._KcConv(mechanism,reaction)
        exparg = self._sortedKcExpArg(mechanism,reaction)
        if conv:
            return conv + ' * exp('+exparg+')'
        else:
            return 'exp('+exparg+')'


    # Set up important QSS variables/containers
    # This function I don't think will be necessary once QSS parsing is implemented. Initialization of QSS networks can be put in setSpecies function
    def _setQSSspecies(self, mechanism, QSS):
        nQSS = len(QSS)

        species = []
        for s in mechanism.species():
            species.append(s.symbol)

        # Get indices of QSS species from species list in mechanism 
        isQSS = np.in1d(species, QSS)
        QSS_index = np.array(np.argwhere(isQSS == True))

        print("\n QSS before sorting: ", QSS, "\n\n" )
        # Sort QSS as they would be found in the mechanism ordering
        QSS_species_index = {}
        QSS_species_sorted = []
        for i in range(nQSS):
            QSS_species_sorted.append(species[int(QSS_index[i])])
            QSS_species_index[species[int(QSS_index[i])]] = i

        # Set global QSS components
        self.nQSS = nQSS
        self.qss_species = QSS_species_sorted
        # QSS_list_index refers to QSS position in QSS_species
        self.QSS_list_index = QSS_species_index
        # QSS_mech_index refers to QSS position in mechanism.species
        self.QSS_mech_index = QSS_index
        # QSS_SSnet and QSS_SRnet are initialized for the entire mechanism
        # and are whittled down to just QSS terms with getQSSsubnetworks() 

        ## ONLY PART OF THIS THAT NEEDS TO BE KEPT: MOVED TO _setSpecies
        self.QSS_SSnet = np.zeros([self.nSpecies, self.nSpecies], 'd')
        self.QSS_SRnet = np.zeros([self.nSpecies, len(mechanism.reaction())], 'd')
        self.QSS_SCnet = np.zeros([nQSS, nQSS], 'd')
        ###########################################################################
        
        # print("\n QSS after sorting: ", self.qss_species,"\n\n")
        # print("\n QSS list index: ", self.QSS_list_index)
        # print("\n QSS mechanism index: ", self.QSS_mech_index)
        # print("QSS_SSnet is \n ", self.QSS_SSnet )


# version
__id__ = "$Id$"

#  End of file 
