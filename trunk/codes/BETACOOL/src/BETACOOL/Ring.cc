
// UAL classes
#include "Main/Teapot.h"
#include "Optics/PacTMap.h"

// Betacool classes
#include "xdynamic.h"

// UAL-Betacool adapters
#include "BETACOOL/Ring.hh"

BETACOOL::Ring::Ring* BETACOOL::Ring::s_theInstance = 0;

BETACOOL::Ring::Ring& BETACOOL::Ring::getInstance(const char* fileName)
{
  if(s_theInstance == 0) {

    std::string fs;
    if(fileName != 0) fs = fileName;

    s_theInstance = new BETACOOL::Ring::Ring(fs);
  }
  return *s_theInstance;
}

BETACOOL::Ring::Ring(std::string& fileName)
{
  char fn[120];
  strcpy(fn, fileName.c_str());
  xData::Get(fn); // read file of parameters and initialization of Betacool objects
  xData::Set(fn); // more initialization of Betacool objects and save file of parameters
}

void BETACOOL::Ring::build(const char* ring)
{
  PacLattices::iterator latIterator = PacLattices::instance()->find(ring);
  if(latIterator == PacLattices::instance()->end()){
    std::cerr << "There is no " << ring << " accelerator " << endl;
    exit(1);
  }

  m_lattice = *latIterator;

  m_teapot.use(m_lattice);

  PAC::BeamAttributes ba;

  PacSurveyData surveyData;
  m_teapot.survey(surveyData);

  double suml = surveyData.survey().suml();

  PAC::Position orbit;
  m_teapot.clorbit(orbit, ba);

  PacTwissData twiss;
  m_teapot.twiss(twiss, ba, orbit);

  std::vector<PacTwissData> vtwiss(m_lattice.size());

  double mux = 0;
  double muy = 0;

  for(int i=0; i < m_lattice.size(); i++) {

    // std::cout << i << " element " << std::endl;

    PacTMap sectorMap(6);
    sectorMap.mltOrder(1);
    sectorMap.refOrbit(orbit);

    m_teapot.trackMap(sectorMap, ba, i, i+1);
    m_teapot.trackClorbit(orbit, ba, i, i+1);

    m_teapot.trackTwiss(twiss, sectorMap);

    if((twiss.mu(0) - mux) < 0.0) twiss.mu(0, twiss.mu(0) + 1.0);
    mux = twiss.mu(0);

    if((twiss.mu(1) - muy) < 0.0) twiss.mu(1, twiss.mu(1) + 1.0);
    muy = twiss.mu(1);

    vtwiss[i] = twiss;
  }

// Start the initialization of Betacool_Ring

  iRing.Number(vtwiss.size()-1);
  iRing.Arc = 0;
  for(int j = 0; j < vtwiss.size()-1; j++){
    iRing[j].EL_LATTICE = true;
    iRing[j].Length = m_lattice[j+1].getLength();
    iRing.Arc += iRing[j].Length;
    iRing[j].Lattice.dist  = iRing.Arc;
    iRing[j].Lattice.betax = vtwiss[j].beta(0);
    iRing[j].Lattice.betay = vtwiss[j].beta(1);
    iRing[j].Lattice.alfax = vtwiss[j].alpha(0);
    iRing[j].Lattice.alfay = vtwiss[j].alpha(1);
    iRing[j].Lattice.Dx = vtwiss[j].d(0);
    iRing[j].Lattice.Dy = vtwiss[j].d(1);
    iRing[j].Lattice.Dpx = vtwiss[j].dp(0);
    iRing[j].Lattice.Dpy = vtwiss[j].dp(1);


  }
  iRing.Circ = iRing.Arc;

}
