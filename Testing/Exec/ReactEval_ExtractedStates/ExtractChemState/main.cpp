
#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Utility.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
#include <AMReX_VisMF.H>
#include <string>
#include <set>
#include <list>

using namespace amrex;
using namespace std;

int main(int argc, char* argv[])
{
  amrex::Initialize(argc,argv);
  {
    ParmParse pp;
#if 1    
    std::string file, fileF;
    pp.get("file",file);
    MultiFab mf;
    VisMF::Read(mf,file);
    const BoxArray& ba = mf.boxArray();
    int nComp = mf.nComp();
    pp.get("fileF",fileF);
    MultiFab mfFt;
    VisMF::Read(mfFt,fileF);
    AMREX_ALWAYS_ASSERT(ba == mfFt.boxArray());
    MultiFab mfF(ba,mf.DistributionMap(),mfFt.nComp(),mfFt.nGrow());
    mfF.copy(mfFt);
    mfFt.clear();
    int nCompF = mfF.nComp();
#else
    Box bx(IntVect(0,0,0),IntVect(127,127,127));
    BoxArray ba(bx);
    ba.maxSize(64);
    DistributionMapping dm(ba);
    int nComp = 4;
    MultiFab mf(ba,dm,nComp,0);
    for (int n=0; n<nComp; ++n) {
      mf.setVal(n,n,1);
    }
#endif
    long seed = 12121965;
    pp.query("seed",seed);
    InitRandom(seed);
    long M = ba.numPts();
    Print() << "BoxArray contains " << M << " cells to sample from" << std::endl;
    long N;
    pp.get("N",N);
    AMREX_ALWAYS_ASSERT(M >= N && N>0);
    set<long> pts;
    std::pair<set<long>::const_iterator,bool> it;
    while (pts.size() < N) {
      it = pts.insert( (long)(Random() * M) );
      while (! it.second )
      {
        it = pts.insert( (long)(Random() * M) );
      }
    }

    Vector<long> offsets(ba.size(),0);
    Vector<set<long>::const_iterator> lboundit(ba.size(),pts.begin());
    for (int i=1; i<ba.size(); ++i) {
      offsets[i] = offsets[i-1] + ba[i-1].numPts();
      lboundit[i] = pts.lower_bound(offsets[i]);
    }

    long nLocal = 0;
    map<int,list<IntVect>> ivs;
    for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
      int n = mfi.index();
      const Box& bx = ba[n];
      const IntVect& ivlo = bx.smallEnd();
      long istride = 1;
      long jstride = bx.length(0);
      long kstride = bx.length(0) * bx.length(1);
      for (auto it = lboundit[n]; it!=pts.end() && it!= lboundit[n+1]; ++it) {
        //AMREX_ALWAYS_ASSERT(*it >= offsets[n] && *it < offsets[n] + bx.numPts());
        long loc = *it - offsets[n];
        int k = int( loc  / kstride );
        int j = int( ( loc - k * kstride ) / jstride );
        int i = int( ( loc - k * kstride - j * jstride ) / istride );
        IntVect iv = ivlo + IntVect(i,j,k);
	if (! bx.contains(iv) ) {
          std::cout << "loc,bx,iv: " << loc << " " << bx << " " << iv << std::endl;
	}
        AMREX_ALWAYS_ASSERT(bx.contains(iv));
        ivs[n].push_back(iv);
      }
      nLocal += ivs[n].size();
    }
    pts.clear();
    lboundit.clear();
    offsets.clear();

    Vector<Real> data(nLocal * (nComp + nCompF));
    long cnt = 0;
    for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
      const auto& fab = mf[mfi];
      const auto& fabF = mfF[mfi];
      const auto& ivlist = ivs[mfi.index()];
      for (list<IntVect>::const_iterator it=ivlist.begin(); it!=ivlist.end(); ++it) {
        long offset = cnt * (nComp + nCompF);
        for (int n=0; n<nComp; ++n) {
          data[offset + n] = fab(*it,n);
	}
        for (int n=0; n<nCompF; ++n) {
          data[offset + nComp + n] = fabF(*it,n);
	}
	cnt++;
      }
    }

    AllPrintToFile ap("junk");
    for (int n=0; n<ParallelDescriptor::NProcs(); ++n) {
      if (n == ParallelDescriptor::MyProc()) {
        for (int m=0; m<nLocal; ++m) {
          stringstream st;
	  st << std::setprecision(20);
          long offset = m * (nComp + nCompF);
          for (int i=0; i<nComp+nCompF; ++i) {
            st << data[offset + i] << " ";
          }
          //std::cout << st.str() << std::endl;
          ap << st.str() << std::endl;
        }
      }
      ParallelDescriptor::Barrier();
    }
  }
  amrex::Finalize();
}

