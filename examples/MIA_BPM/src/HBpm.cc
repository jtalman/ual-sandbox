
#include "UAL/APF/PropagatorFactory.hh"
#include "BPMCollector.hh"
#include "HBpm.hh"


MIA::HBpm::HBpm() : MIA::BPM()
{
  m_type = "h";
}

MIA::HBpm::HBpm(const MIA::HBpm& bpm) : MIA::BPM(bpm)
{
}

MIA::HBpm::~HBpm()
{
}

UAL::PropagatorNode* MIA::HBpm::clone()
{
  return new MIA::HBpm::HBpm(*this);
}

void MIA::HBpm::setLatticeElements(const UAL::AcceleratorNode& sequence, 
				  int is0, 
				  int is1,
				  const UAL::AttributeSet& attSet)
{

  init(sequence, is0, is1, attSet);

  // std::cout << "BPM i=" << m_i << ", name = " << m_name 
  // 	    << ", design name = " << m_designName << std::endl;

  MIA::BPMCollector::getInstance().registerBPM(this);
}

void MIA::HBpm::write(std::ofstream& out)
{
  out << "BPM name: "   << m_name << std::endl;
  out << "BPM scoord: " << m_s    << std::endl;
  out << "BPM type: "   << m_type << std::endl;

  int turn = 0;
  for(std::list<PAC::Position>::iterator it = m_data.begin(); it != m_data.end(); it++){
    out << turn++ << " " << it->getX() << std::endl;
  }
}


MIA::HBpmRegister::HBpmRegister()
{
  UAL::PropagatorNodePtr nodePtr(new MIA::HBpm());
  UAL::PropagatorFactory::getInstance().add("MIA::HBpm", nodePtr);
}

static MIA::HBpmRegister the_MIA_HBPM_Register; 
