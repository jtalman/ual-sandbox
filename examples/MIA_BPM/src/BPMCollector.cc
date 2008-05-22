
#include "BPMCollector.hh"

MIA::BPMCollector* MIA::BPMCollector::s_theInstance = 0;

MIA::BPMCollector::BPMCollector()
{
  m_hBPMs  = 0;
  m_vBPMs  = 0;
  m_hvBPMs = 0;
}

MIA::BPMCollector& MIA::BPMCollector::getInstance()
{
  if(s_theInstance == 0){
    s_theInstance = new MIA::BPMCollector();
  }
  return *s_theInstance;
}

void MIA::BPMCollector::registerBPM(MIA::BPM* bpm)
{
  std::map<int, MIA::BPM*>::iterator it = m_bpms.find(bpm->getIndex());
  if(it != m_bpms.end()){
    std::cout << "BPM " << bpm->getName() << " has been already registered " << std::endl;
    return;
  }

  m_bpms[bpm->getIndex()] = bpm;

  switch(bpm->getType()[0]){
  case 'h':
    m_hBPMs++;
    break;
  case 'v':
    m_vBPMs++;
    break;
  default:
    m_hvBPMs++;
  };
}

std::map<int, MIA::BPM*>&  MIA::BPMCollector::getAllData()
{
  return m_bpms;
}


void MIA::BPMCollector::clear()
{
  std::map<int, MIA::BPM*>::iterator ibpm;
  for(ibpm = m_bpms.begin(); ibpm != m_bpms.end(); ibpm++){
    ibpm->second->clear();
  }
}

void MIA::BPMCollector::write(const char* fileName)
{


  m_file.open(fileName);
  if(!m_file) {
    std::cerr << "Cannot open BPMCollector output file " << fileName << std::endl;
    return;
  }

  if(m_bpms.begin() == m_bpms.end()) {
    m_file.close();
    return;
  }

  m_file << "lattice: " << m_bpms.begin()->second->getLatticeName() << std::endl;
  m_file << "hBPMs: "   << m_hBPMs  << std::endl;
  m_file << "vBPMs: "   << m_vBPMs  << std::endl;
  m_file << "hvBPMs: "  << m_hvBPMs << std::endl;
  m_file << "nTurns: "  << m_bpms.begin()->second->getTurns() << std::endl;
 
  m_file << std::endl;

  int counter = 0;
  std::map<int, MIA::BPM*>::iterator ibpm;
  for(ibpm = m_bpms.begin(); ibpm != m_bpms.end(); ibpm++){

    m_file <<  "BPM index : " << counter++ << std::endl;
    ibpm->second->write(m_file); 
    m_file << "\n\n";

  }

  m_file.close();

}

