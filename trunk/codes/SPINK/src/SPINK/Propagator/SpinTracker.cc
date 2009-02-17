// Library       : SPINK
// File          : SPINK/SpinMapper/SpinMapper.cc
// Copyright     : see Copyright file
// Author        : A.Luccio
// C++ version   : N.Malitsky 

#include "PAC/Beam/Bunch.hh"
#include "TEAPOT/Integrator/TrackerFactory.hh"
#include "SPINK/Propagator/SpinTracker.hh"

std::vector<PAC::Position> SPINK::SpinTracker::s_positions(64);

SPINK::SpinTracker::SpinTracker()
{

}

SPINK::SpinTracker::~SpinTracker()
{
}


void SPINK::SpinTracker::setLatticeElements(const UAL::AcceleratorNode& sequence,
					   int is0, int is1,
					   const UAL::AttributeSet& attSet)
{
    SPINK::SpinPropagator::setLatticeElements(sequence, is0, is1, attSet);
 
    const PacLattice& lattice      = (PacLattice&) sequence;
    UAL::PropagatorNodePtr nodePtr = 
      TEAPOT::TrackerFactory::createTracker(lattice[is0].getType());
    m_tracker = nodePtr;

    m_tracker->setLatticeElements(sequence, is0, is1, attSet);
}

void SPINK::SpinTracker::propagate(UAL::Probe& b)
{
  PAC::Bunch& bunch = static_cast<PAC::Bunch&>(b);
  m_tracker->propagate(bunch);
}


