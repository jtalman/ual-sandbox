// Library       : SIMBAD
// File          : SIMBAD/SC/TSCPropagatorFFT.cc
// Copyright     : see Copyright file
// Author        : N.D'Imperio, A.Luccio et al.

#include "UAL/APF/PropagatorFactory.hh"
#include "PAC/Beam/Bunch.hh"
#include "TEAPOT/Integrator/TrackerFactory.hh"
#include "SIMBAD/Tracker/MatrixTracker.hh"
#include "SIMBAD/SC_MPI/TSCPropagatorFFT_MPI.hh"
#include "SIMBAD/SC_MPI/TSCCalculatorFFT_MPI.hh"

using namespace std;


SIMBAD::TSCPropagatorFFT_MPI::TSCPropagatorFFT_MPI()
{
}

SIMBAD::TSCPropagatorFFT_MPI::TSCPropagatorFFT_MPI(const SIMBAD::TSCPropagatorFFT_MPI& p)
{
}

SIMBAD::TSCPropagatorFFT_MPI::~TSCPropagatorFFT_MPI()
{
}

UAL::PropagatorNode* SIMBAD::TSCPropagatorFFT_MPI::clone()
{
  return new SIMBAD::TSCPropagatorFFT_MPI(*this);
}




void SIMBAD::TSCPropagatorFFT_MPI::propagate(UAL::Probe& b)
{
  PAC::Bunch& bunch = static_cast<PAC::Bunch&>(b);
  if(m_tracker.isValid()) m_tracker->propagate(bunch);

  SIMBAD::TSCCalculatorFFT_MPI::getInstance().calculateForce(bunch);
  SIMBAD::TSCCalculatorFFT_MPI::getInstance().propagate(bunch, m_lkick);
}

SIMBAD::TSCPropagatorFFT_MPIRegister::TSCPropagatorFFT_MPIRegister()
{
  UAL::PropagatorNodePtr nodePtr(new SIMBAD::TSCPropagatorFFT_MPI());
  UAL::PropagatorFactory::getInstance().add("SIMBAD::TSCPropagatorFFT_MPI", nodePtr);
}

static SIMBAD::TSCPropagatorFFT_MPIRegister theSimbadTSCPropagatorFFT_MPIRegister; 


