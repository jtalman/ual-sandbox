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

    m_l = lattice[is0].getLength();
    m_n = lattice[is0].getN();

    UAL::PropagatorNodePtr nodePtr = 
      TEAPOT::TrackerFactory::createTracker(lattice[is0].getType());
    m_tracker = nodePtr;

    if(m_n) {

        PacLattElement* le = const_cast<PacLattElement*>(&lattice[is0]);

        int ns = m_n*4;
        double dl = m_l - m_l/ns;

        le->addLength(-dl);
        le->addN(-m_n);

        m_tracker->setLatticeElements(sequence, is0, is1, attSet);
        
        le->addLength(dl);
        le->addN(m_n);

    }

    std::cout << lattice[is0].getName() << " " << m_l << " " << m_n << std::endl;
}

void SPINK::SpinTracker::propagate(UAL::Probe& b)
{
  PAC::Bunch& bunch = static_cast<PAC::Bunch&>(b);

  if(m_n == 0){
      m_tracker->propagate(bunch);
      // add spink
      return;
  }

  int ns = m_n*4; // number of slices

  for(int i=0; i < ns; i++) {
    m_tracker->propagate(bunch);
    // add spink
  }
  
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



