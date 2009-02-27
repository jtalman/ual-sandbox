// Library       : SPINK
// File          : SPINK/SpinMapper/SpinMapper.cc
// Copyright     : see Copyright file
// Author        : A.Luccio
// C++ version   : N.Malitsky 

#include "UAL/APF/PropagatorFactory.hh"
#include "PAC/Beam/Bunch.hh"
#include "TEAPOT/Integrator/TrackerFactory.hh"
#include "SPINK/Propagator/SpinTracker.hh"

std::vector<PAC::Position> SPINK::SpinTracker::s_positions(64);

SPINK::SpinTracker::SpinTracker()
{

}

SPINK::SpinTracker::SpinTracker(const SPINK::SpinTracker& st)
{
  copy(st);
}

SPINK::SpinTracker::~SpinTracker()
{
}

UAL::PropagatorNode* SPINK::SpinTracker::clone()
{
  return new SPINK::SpinTracker(*this);
}


void SPINK::SpinTracker::setLatticeElements(const UAL::AcceleratorNode& sequence,
					   int is0, int is1,
					   const UAL::AttributeSet& attSet)
{
    SPINK::SpinPropagator::setLatticeElements(sequence, is0, is1, attSet);
 
    const PacLattice& lattice      = (PacLattice&) sequence;

    std::cout << is0 << " " << lattice[is0].getName() << " " << lattice[is0].getType() << std::endl;

    UAL::PropagatorNodePtr nodePtr = 
      TEAPOT::TrackerFactory::createTracker(lattice[is0].getType());
    m_tracker = nodePtr;

    m_tracker->setLatticeElements(sequence, is0, is1, attSet);

    TEAPOT::BasicTracker* bt =
                static_cast<TEAPOT::BasicTracker*>(m_tracker.getPointer());

    m_l  = bt->getLength();
    m_n = bt->getN();

    std::cout << lattice[is0].getName() << " " << m_l << " " << m_n << std::endl;
}

void SPINK::SpinTracker::propagate(UAL::Probe& b)
{
  PAC::Bunch& bunch = static_cast<PAC::Bunch&>(b);

  // for (int is=0; is < nslices; is++) {
  m_tracker->propagate(bunch);
  
  /*
  int size = bunch.size();
  for(int i=0; i < size; i++){
   if(bunch[i].isLost() ) continue;

    PAC::Position& pos = bunch[i].getPosition();

    x0  = pos.getX();
    px0 = pos.getPX();
    y0  = pos.getY();
    py0 = pos.getPY();
    ct0 = pos.getCT();
    de0 = pos.getDE();

   PAC::Spin& spin = bunch[i].getSpin();
   
   ...
   
   kls(3) -> take it

  }
   */
// }

}

void SPINK::SpinTracker::copy(const SPINK::SpinTracker& st)
{
    m_l = st.m_l;
    m_n = st.m_n;
}

SPINK::SpinTrackerRegister::SpinTrackerRegister()
{
  UAL::PropagatorNodePtr driftPtr(new SPINK::SpinTracker());
  UAL::PropagatorFactory::getInstance().add("SPINK::SpinTracker", driftPtr);
}

static SPINK::SpinTrackerRegister theSpinkSpinTrackerRegister;



