// Library       : SPINK
// File          : SPINK/Propagator/DriftTracker.cc
// Copyright     : see Copyright file
// Author        : A.Luccio
// C++ version   : N.Malitsky

#include "UAL/APF/PropagatorFactory.hh"
#include "PAC/Beam/Bunch.hh"
#include "TEAPOT/Integrator/BasicTracker.hh"
#include "TEAPOT/Integrator/TrackerFactory.hh"
#include "SPINK/Propagator/DriftTracker.hh"


SPINK::DriftTracker::DriftTracker()
{

}

SPINK::DriftTracker::DriftTracker(const SPINK::DriftTracker& dt)
        : SPINK::SpinTracker(dt)
{
}

SPINK::DriftTracker::~DriftTracker()
{
}

UAL::PropagatorNode* SPINK::DriftTracker::clone()
{
  return new SPINK::DriftTracker(*this);
}


void SPINK::DriftTracker::setLatticeElements(const UAL::AcceleratorNode& sequence,
					   int is0, int is1,
					   const UAL::AttributeSet& attSet)
{
    SPINK::SpinTracker::setLatticeElements(sequence, is0, is1, attSet);

}

void SPINK::DriftTracker::propagate(UAL::Probe& b)
{
  PAC::Bunch& bunch = static_cast<PAC::Bunch&>(b);
  m_tracker->propagate(bunch);
}



SPINK::DriftTrackerRegister::DriftTrackerRegister()
{
  UAL::PropagatorNodePtr driftPtr(new SPINK::DriftTracker());
  UAL::PropagatorFactory::getInstance().add("SPINK::DriftTracker", driftPtr);
}

static SPINK::DriftTrackerRegister theSpinkDriftTrackerRegister;




