
#include "UAL/APF/PropagatorFactory.hh"
#include "PAC/Beam/Bunch.hh"
#include "SMF/PacLattice.h"
#include "BPM.hh"
#include "BPMCollector.hh"


MIA::BPM::BPM() 
{
  init();
  m_type = "x";
}

MIA::BPM::BPM(const MIA::BPM& bpm)
{
  copy(bpm);
}

MIA::BPM::~BPM()
{
}

UAL::PropagatorNode* MIA::BPM::clone()
{
  return new MIA::BPM::BPM(*this);
}

UAL::AcceleratorNode& MIA::BPM::getFrontAcceleratorNode()
{
  return m_frontNode;
}

UAL::AcceleratorNode& MIA::BPM::getBackAcceleratorNode()
{
  return m_backNode;
}

void MIA::BPM::setLatticeElements(const UAL::AcceleratorNode& sequence, 
				  int is0, 
				  int is1,
				  const UAL::AttributeSet& attSet)
{

  init(sequence, is0, is1, attSet);

  // std::cout << "BPM i=" << m_i << ", name = " << m_name 
  // << ", design name = " << m_designName << std::endl;

  MIA::BPMCollector::getInstance().registerBPM(this);
}

void MIA::BPM::propagate(UAL::Probe& probe)
{
  PAC::Bunch& bunch = static_cast<PAC::Bunch&>(probe);
  m_data.push_back(bunch[0].getPosition());
}

void MIA::BPM::write(std::ofstream& out)
{
  out << "BPM name: "   << m_name << std::endl;
  out << "BPM scoord: " << m_s    << std::endl;
  out << "BPM type: "   << "hv"   << std::endl;

  int turn = 0;
  for(std::list<PAC::Position>::iterator it = m_data.begin(); it != m_data.end(); it++){
    out << turn++ << " " << it->getX() << " " << it->getY() << std::endl;
  }
}

void MIA::BPM::clear()
{
  m_data.clear();
}

int MIA::BPM::getIndex() const 
{
  return m_i;
}

const std::string&  MIA::BPM::getName() const 
{
  return m_name;
}

const std::string&  MIA::BPM::getType() const 
{
  return m_type;
}

void MIA::BPM::init()
{
  m_i = -1;
  m_s = 0.0;
}

void MIA::BPM::copy(const MIA::BPM& bpm)
{
  m_accName           = bpm.m_accName;
  m_name              = bpm.m_name;
  m_type              = bpm.m_type;
  m_designName        = bpm.m_designName;
  m_i                 = bpm.m_i;
  m_s                 = bpm.m_s;
  m_data              = bpm.m_data;
}

void MIA::BPM::init(const UAL::AcceleratorNode& sequence, 
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


MIA::BPMRegister::BPMRegister()
{
  UAL::PropagatorNodePtr nodePtr(new MIA::BPM());
  UAL::PropagatorFactory::getInstance().add("MIA::BPM", nodePtr);
}

static MIA::BPMRegister the_MIA_BPM_Register; 
