

#include "UAL/APF/PropagatorFactory.hh"
#include "PAC/Beam/Bunch.hh"
#include "SMF/PacLattice.h"
#include "AIM/BPM/BPM.hh"
#include "AIM/BPM/MonitorCollector.hh"


AIM::BPM::BPM() 
{
  init();
}

AIM::BPM::BPM(const AIM::BPM& bpm)
{
  copy(bpm);
}

AIM::BPM::~BPM()
{
}

UAL::PropagatorNode* AIM::BPM::clone()
{
  return new AIM::BPM::BPM(*this);
}

UAL::AcceleratorNode& AIM::BPM::getFrontAcceleratorNode()
{
  return m_frontNode;
}

UAL::AcceleratorNode& AIM::BPM::getBackAcceleratorNode()
{
  return m_backNode;
}

void AIM::BPM::setLatticeElements(const UAL::AcceleratorNode& sequence, 
				  int is0, 
				  int is1,
				  const UAL::AttributeSet& attSet)
{

  init(sequence, is0, is1, attSet);

  // std::cout << "BPM i=" << m_i << ", name = " << m_name 
  // << ", design name = " << m_designName << std::endl;

  AIM::MonitorCollector::getInstance().registerBPM(this);
}

void AIM::BPM::propagate(UAL::Probe& probe)
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

void AIM::BPM::write(std::ofstream& out)
{
  out << "BPM name: "   << m_name << std::endl;
  out << "BPM scoord: " << m_s    << std::endl;
  out << "BPM type: "   << "hv"   << std::endl;

  int turn = 0;
  for(std::list<PAC::Position>::iterator it = m_data.begin(); it != m_data.end(); it++){
    out << turn++ << " " << it->getX() << " " << it->getY() << std::endl;
  }
}

void AIM::BPM::clear()
{
  m_data.clear();
}

int AIM::BPM::getIndex() const 
{
  return m_i;
}

const std::string&  AIM::BPM::getName() const 
{
  return m_name;
}

const std::string&  AIM::BPM::getType() const 
{
  return m_type;
}

void AIM::BPM::init()
{
  m_i = -1;
  m_s = 0.0;
  m_type = "x";
}

void AIM::BPM::copy(const AIM::BPM& bpm)
{
  m_accName           = bpm.m_accName;
  m_name              = bpm.m_name;
  m_type              = bpm.m_type;
  m_designName        = bpm.m_designName;
  m_i                 = bpm.m_i;
  m_s                 = bpm.m_s;
  m_data              = bpm.m_data;
}

void AIM::BPM::init(const UAL::AcceleratorNode& sequence, 
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

AIM::BPMRegister::BPMRegister()
{
  UAL::PropagatorNodePtr nodePtr(new AIM::BPM());
  UAL::PropagatorFactory::getInstance().add("AIM::BPM", nodePtr);
}

static AIM::BPMRegister the_AIM_BPM_Register; 
