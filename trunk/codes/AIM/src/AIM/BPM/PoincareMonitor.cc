#include <fstream>

#include "UAL/APF/PropagatorFactory.hh"
#include "PAC/Beam/Bunch.hh"
#include "SMF/PacLattice.h"

#include "AIM/BPM/PoincareMonitor.hh"
#include "AIM/BPM/PoincareMonitorCollector.hh"


AIM::PoincareMonitor::PoincareMonitor() 
{
  init();
}

AIM::PoincareMonitor::PoincareMonitor(const AIM::PoincareMonitor& monitor)
{
  copy(monitor);
}

AIM::PoincareMonitor::~PoincareMonitor()
{
}

UAL::PropagatorNode* AIM::PoincareMonitor::clone()
{
  return new AIM::PoincareMonitor::PoincareMonitor(*this);
}

UAL::AcceleratorNode& AIM::PoincareMonitor::getFrontAcceleratorNode()
{
  return m_frontNode;
}

UAL::AcceleratorNode& AIM::PoincareMonitor::getBackAcceleratorNode()
{
  return m_backNode;
}

void AIM::PoincareMonitor::setLatticeElements(const UAL::AcceleratorNode& sequence, 
					      int is0, 
					      int is1,
					      const UAL::AttributeSet& attSet)
{

  init(sequence, is0, is1, attSet);

  // std::cout << "BPM i=" << m_i << ", name = " << m_name 
  // << ", design name = " << m_designName << std::endl;

  AIM::PoincareMonitorCollector& mc = AIM::PoincareMonitorCollector::getInstance();
  if(mc.getAllData().size() > 0) return;

  std::cout << "PoincareMonitor: registered  " << m_designName << std::endl;
  AIM::PoincareMonitorCollector::getInstance().registerBPM(this);
}

void AIM::PoincareMonitor::propagate(UAL::Probe& probe)
{
  PAC::Bunch bunch = static_cast<PAC::Bunch&>(probe);  
  m_data.push_back(bunch);
}



void AIM::PoincareMonitor::clear()
{
  m_data.clear();
}

int AIM::PoincareMonitor::getIndex() const 
{
  return m_i;
}

const std::string&  AIM::PoincareMonitor::getName() const 
{
  return m_name;
}



void AIM::PoincareMonitor::init()
{
  m_i = -1;
  m_s = 0.0;
}

void AIM::PoincareMonitor::copy(const AIM::PoincareMonitor& bpm)
{
  m_accName           = bpm.m_accName;
  m_name              = bpm.m_name;
  m_designName        = bpm.m_designName;
  m_i                 = bpm.m_i;
  m_s                 = bpm.m_s;
  m_data              = bpm.m_data;
}

void AIM::PoincareMonitor::init(const UAL::AcceleratorNode& sequence, 
		    int is0, 
		    int is1,
		    const UAL::AttributeSet& attSet)
{
   if(is0 < sequence.getNodeCount()) m_frontNode = 
				       *((PacLattElement*) sequence.getNodeAt(is0));
   if(is1 < sequence.getNodeCount()) m_backNode  = 
				       *((PacLattElement*) sequence.getNodeAt(is1));

   const PacLattice& lattice     = (PacLattice&) sequence;

   m_accName               = lattice.getName();
   m_name                  = lattice[is0].getName();
   m_designName            = lattice[is0].getDesignName();
   m_i                     = is0;
   m_s                     = lattice[is0].getPosition();

}

void AIM::PoincareMonitor::write(const char* fileName)
{
  std::ofstream out(fileName);

  out << "PoincareMonitor"  
      << ", design name: " << m_designName 
      << ", index: " << m_i << std::endl;

  int turn = 0;
  std::list<PAC::Bunch>::iterator it;
  for (it =  m_data.begin(); it != m_data.end(); it++){

    out << "turn = " << turn << std::endl;
    PAC::Bunch& bunch = *it;

    for(int i = 0; i < bunch.size(); i++){

      if(bunch[i].isLost()) continue;

      PAC::Position& pos = bunch[i].getPosition();
      out << i << " " 
	  << pos.getX() << " " 
	  << pos.getPX() << " " 
	  << pos.getY() << " " 
	  << pos.getPY() << " " 
	  << pos.getCT() << " " 
	  << pos.getDE() << std::endl;
    }

    turn++;
  }

  out.close();
}

AIM::PoincareMonitorRegister::PoincareMonitorRegister()
{
  UAL::PropagatorNodePtr nodePtr(new AIM::PoincareMonitor());
  UAL::PropagatorFactory::getInstance().add("AIM::PoincareMonitor", nodePtr);
}

static AIM::PoincareMonitorRegister the_AIM_PoincareMonitor_Register; 
