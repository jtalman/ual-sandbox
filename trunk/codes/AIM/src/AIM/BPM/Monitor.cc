

#include "UAL/APF/PropagatorFactory.hh"
#include "PAC/Beam/Bunch.hh"
#include "SMF/PacLattice.h"

#include "AIM/BPM/Monitor.hh"
#include "AIM/BPM/MonitorCollector.hh"


AIM::Monitor::Monitor() 
{
  init();
}

AIM::Monitor::Monitor(const AIM::Monitor& monitor)
{
  copy(monitor);
}

AIM::Monitor::~Monitor()
{
}

UAL::PropagatorNode* AIM::Monitor::clone()
{
  return new AIM::Monitor::Monitor(*this);
}

UAL::AcceleratorNode& AIM::Monitor::getFrontAcceleratorNode()
{
  return m_frontNode;
}

UAL::AcceleratorNode& AIM::Monitor::getBackAcceleratorNode()
{
  return m_backNode;
}

void AIM::Monitor::setLatticeElements(const UAL::AcceleratorNode& sequence, 
				  int is0, 
				  int is1,
				  const UAL::AttributeSet& attSet)
{

  init(sequence, is0, is1, attSet);

  // std::cout << "BPM i=" << m_i << ", name = " << m_name 
  // << ", design name = " << m_designName << std::endl;

  AIM::MonitorCollector::getInstance().registerBPM(this);
}

void AIM::Monitor::propagate(UAL::Probe& probe)
{
  PAC::Bunch& bunch = static_cast<PAC::Bunch&>(probe);
  
  int counter = 0;
  PAC::Position pos;
  for(int i=0; i < bunch.size(); i++){
    if(bunch[i].isLost()) continue;
    pos += bunch[i].getPosition();
    counter++;
  }

  pos /= counter;    
  m_data.push_back(pos);
}

void AIM::Monitor::write(std::ofstream& out)
{
  out << "BPM name: "   << m_name << std::endl;
  out << "BPM scoord: " << m_s    << std::endl;
  out << "BPM type: "   << "hv"   << std::endl;

  int turn = 0;
  for(std::list<PAC::Position>::iterator it = m_data.begin(); it != m_data.end(); it++){
    out << turn++ << " " << it->getX() << " " << it->getY() << std::endl;
  }
}

void AIM::Monitor::clear()
{
  m_data.clear();
}

int AIM::Monitor::getIndex() const 
{
  return m_i;
}

const std::string&  AIM::Monitor::getName() const 
{
  return m_name;
}

const std::string&  AIM::Monitor::getType() const 
{
  return m_type;
}

void AIM::Monitor::init()
{
  m_i = -1;
  m_s = 0.0;
  m_type = "x";
}

void AIM::Monitor::copy(const AIM::Monitor& bpm)
{
  m_accName           = bpm.m_accName;
  m_name              = bpm.m_name;
  m_type              = bpm.m_type;
  m_designName        = bpm.m_designName;
  m_i                 = bpm.m_i;
  m_s                 = bpm.m_s;
  m_data              = bpm.m_data;
}

void AIM::Monitor::init(const UAL::AcceleratorNode& sequence, 
		    int is0, 
		    int is1,
		    const UAL::AttributeSet& attSet)
{
   if(is0 < sequence.getNodeCount()) m_frontNode = *((PacLattElement*) sequence.getNodeAt(is0));
   if(is1 < sequence.getNodeCount()) m_backNode  = *((PacLattElement*) sequence.getNodeAt(is1));

   const PacLattice& lattice     = (PacLattice&) sequence;

   m_accName               = lattice.getName();
   m_name                  = lattice[is0].getName();
   m_designName            = lattice[is0].getDesignName();
   m_i                     = is0;
   m_s                     = lattice[is0].getPosition();

}

AIM::MonitorRegister::MonitorRegister()
{
  UAL::PropagatorNodePtr nodePtr(new AIM::Monitor());
  UAL::PropagatorFactory::getInstance().add("AIM::Monitor", nodePtr);
}

static AIM::MonitorRegister the_AIM_MONITOR_Register; 
