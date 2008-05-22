// Library       : SIMBAD
// File          : SIMBAD/SC/TSCPropagatorFFT_3D_MPI.cc
// Copyright     : see Copyright file
// Author        : N.D'Imperio, A.Luccio et al.

#include "UAL/APF/PropagatorFactory.hh"
#include "PAC/Beam/Bunch.hh"
#include "TEAPOT/Integrator/TrackerFactory.hh"
#include "SIMBAD/Tracker/MatrixTracker.hh"
#include "SIMBAD/SC3D_MPI/TSCPropagatorFFT_3D_MPI.hh"
#include "SIMBAD/SC3D_MPI/TSCCalculatorFFT_3D_MPI.hh"

using namespace std;

SIMBAD::TSCPropagatorFFT_3D_MPI::TSCPropagatorFFT_3D_MPI()
{
}

SIMBAD::TSCPropagatorFFT_3D_MPI::TSCPropagatorFFT_3D_MPI(const SIMBAD::TSCPropagatorFFT_3D_MPI& p)
{
}

SIMBAD::TSCPropagatorFFT_3D_MPI::~TSCPropagatorFFT_3D_MPI()
{
}

UAL::PropagatorNode* SIMBAD::TSCPropagatorFFT_3D_MPI::clone()
{
  return new SIMBAD::TSCPropagatorFFT_3D_MPI(*this);
}


void SIMBAD::TSCPropagatorFFT_3D_MPI::propagate(UAL::Probe& b)
{
  PAC::Bunch& bunch = static_cast<PAC::Bunch&>(b);
  if(m_tracker.isValid()) m_tracker->propagate(bunch);

  vector<vector<int> > subBunchIndicesVect;

  SIMBAD::LoadBalancer& lb = SIMBAD::LoadBalancer::getInstance(bunch);
  
  lb.exchangeParticles(bunch); // MPI

  lb.assignMacrosToSB(bunch, subBunchIndicesVect);


  SIMBAD::TSCCalculatorFFT_3D_MPI& calculator =
    SIMBAD::TSCCalculatorFFT_3D_MPI::getInstance();

  for(int i = lb.getStartSB(); i < lb.getStartSB()+lb.getLocalNSB(); i++)
    {
      calculator.calculateForce(bunch, subBunchIndicesVect[i]);
      calculator.propagate(bunch, subBunchIndicesVect[i], m_lkick);
    }

  
}


SIMBAD::TSCPropagatorFFT_3D_MPIRegister::TSCPropagatorFFT_3D_MPIRegister()
{
  UAL::PropagatorNodePtr nodePtr(new SIMBAD::TSCPropagatorFFT_3D_MPI());
  UAL::PropagatorFactory::getInstance().add("SIMBAD::TSCPropagatorFFT_3D_MPI", nodePtr);
}

static SIMBAD::TSCPropagatorFFT_3D_MPIRegister theSimbadTSCPropagatorFFT_3D_MPIRegister;

