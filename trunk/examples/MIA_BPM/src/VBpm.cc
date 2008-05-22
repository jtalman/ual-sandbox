
#include "UAL/APF/PropagatorFactory.hh"
#include "BPMCollector.hh"
#include "VBpm.hh"


MIA::VBpm::VBpm() : MIA::BPM()
{
  m_type = "v";
}

MIA::VBpm::VBpm(const MIA::VBpm& bpm) : MIA::BPM(bpm)
{
}

MIA::VBpm::~VBpm()
{
}

UAL::PropagatorNode* MIA::VBpm::clone()
{
  return new MIA::VBpm::VBpm(*this);
}

void MIA::VBpm::setLatticeElements(const UAL::AcceleratorNode& sequence, 
				  int is0, 
				  int is1,
				  const UAL::AttributeSet& attSet)
{

  init(sequence, is0, is1, attSet);

  // std::cout << "BPM i=" << m_i << ", name = " << m_name 
  // 	    << ", design name = " << m_designName << std::endl;

  MIA::BPMCollector::getInstance().registerBPM(this);
}

void MIA::VBpm::write(std::ofstream& out)
{
  out << "BPM name: "   << m_name << std::endl;
  out << "BPM scoord: " << m_s    << std::endl;
  out << "BPM type: "   << m_type << std::endl;

  int turn = 0;
  for(std::list<PAC::Position>::iterator it = m_data.begin(); it != m_data.end(); it++){
    out << turn++ << " " << it->getY() << std::endl;
  }
}


MIA::VBpmRegister::VBpmRegister()
{
  UAL::PropagatorNodePtr nodePtr(new MIA::VBpm());
  UAL::PropagatorFactory::getInstance().add("MIA::VBpm", nodePtr);
}

static MIA::VBpmRegister the_MIA_HBPM_Register; 
