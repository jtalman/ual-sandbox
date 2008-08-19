#include <ctype.h>
#include <regex.h> 

#include "UAL/Common/Def.hh"
#include "Optics/PacChromData.h"
#include "PAC/Beam/Bunch.hh"
#include "Optics/PacTMap.h"

#include "UAL/UI/ShellImp.hh"
#include "UAL/UI/OpticsCalculator.hh"

#include "Main/Teapot.h"

UAL::OpticsCalculator* UAL::OpticsCalculator::s_theInstance = 0;

UAL::OpticsCalculator& UAL::OpticsCalculator::getInstance()
{
  if(s_theInstance == 0){
    s_theInstance = new UAL::OpticsCalculator();
  }
  return *s_theInstance;
}

UAL::OpticsCalculator::OpticsCalculator()
{
  m_teapot = 0; // new Teapot();
  m_chrom  = new PacChromData();
}

UAL::OpticsCalculator::~OpticsCalculator()
{
  if(m_teapot) delete m_teapot;
}

void UAL::OpticsCalculator::setBeamAttributes(const PAC::BeamAttributes& ba)
{
  m_ba = ba;
}

bool UAL::OpticsCalculator::use(const std::string& accName){

  PacLattices::iterator latIterator = PacLattices::instance()->find(accName);
  if(latIterator == PacLattices::instance()->end()){
    std::cerr << "OpticsCalculator: There is no " + accName << " accelerator " << endl;
    return false;
  }

  if(m_teapot) delete m_teapot;
  m_teapot = new Teapot();
  m_teapot->use(*latIterator);
}



bool UAL::OpticsCalculator::calculate()
{
  if(!m_teapot) return false;
  if(m_teapot->size() == 0) return false;

  PacSurveyData surveyData;
  m_teapot->survey(surveyData);

  suml = surveyData.survey().suml();

  // std::cout << "UAL::OpticsCalculator::calculate(): suml = " << suml << std::endl;

  PAC::Position orbit;
  m_teapot->clorbit(orbit, m_ba);

  // std::cout << "UAL::OpticsCalculator::calculate(): closed orbit " << std::endl;

  PacChromData chr;
  m_teapot->chrom(chr, m_ba, orbit);

  *m_chrom = chr;

  PAC::Bunch bunch(1);
  bunch.getBeamAttributes().setRevfreq(UAL::clight/suml);

  bunch[0].getPosition().set(0.0, 0.0, 0.0, 0.0, 0.0,  1.e-5);

  m_teapot->track(bunch);

  double ct0 =  bunch[0].getPosition().getCT();
  double de0 =  bunch[0].getPosition().getDE();

  alpha0 = (-ct0/UAL::clight)*bunch.getBeamAttributes().getRevfreq()/de0; 

  // std::cout << "Optics Calculator: gt = " << 1/sqrt(abs(alpha0)) << std::endl;

  return true;
}

void UAL::OpticsCalculator::getOrbit(std::vector<double>& positions,
				     std::vector<PAC::Position>& orbit)
{
  if(!m_teapot) return;
  std::vector<int> elems(m_teapot->size());
  for(int i = 0; i < m_teapot->size(); i++){
    elems[i] = i;
  }

  calculatePositions(elems, positions);

  orbit.clear();
  calculateOrbit(elems, orbit);

}

void UAL::OpticsCalculator::getTwiss(std::vector<double>& positions,
				     std::vector<PacTwissData>& twiss)
{
  if(!m_teapot) return;
  std::vector<int> elems(m_teapot->size());
  for(int i = 0; i < m_teapot->size(); i++){
    elems[i] = i;
  }

  calculatePositions(elems, positions);

  std::vector<PacVTps> maps;
  calculateMaps(elems, maps, 1);

  twiss.clear();
  calculateTwiss(elems, maps, twiss);

}

void UAL::OpticsCalculator::calculatePositions(const std::vector<int>& elems,
					       std::vector<double>& positions)
{
  if(!m_teapot) return;
  positions.resize(elems.size());

  PacSurveyData surveyData; 

  int i1 = 0, i2;
  for(unsigned int i = 0; i < elems.size(); i++){

    i2 = elems[i];
 
    m_teapot->survey(surveyData, i1, i2);

    positions[i] = surveyData.survey().suml();
    i1 = i2;
  }
}

void UAL::OpticsCalculator::calculateMaps(const std::vector<int>& elems,
					  std::vector<PacVTps>& maps,
					  int order)
{
  if(!m_teapot) return;

  PAC::Position orbit;
  m_teapot->clorbit(orbit, m_ba);

  PacTMap map(6);
  map.refOrbit(orbit);
  m_teapot->map(map, m_ba, order);

  int mltOrder =  map.mltOrder();
  map.mltOrder(order);

  int i1 = 0, i2;
  maps.resize(elems.size());
  for(unsigned int it = 0; it < elems.size(); it++){

    i2 = elems[it];

    // std::cout << "maps " << it << " : (" << i1 << ", " << i2 << ")" << std::endl;

    PacTMap map(6);
    map.refOrbit(orbit);

    m_teapot->trackMap(map, m_ba, i1, i2);
    maps[it] = static_cast<PacVTps&>(map);

    m_teapot->trackClorbit(orbit, m_ba, i1, i2);

    i1 = i2;    
  }

  map.mltOrder(mltOrder);

}

void UAL::OpticsCalculator::calculateOrbit(const std::vector<int>& elems,
					   std::vector<PAC::Position>& orbitVector)
{
  if(!m_teapot) return;


  PAC::Position orbit;
  m_teapot->clorbit(orbit, m_ba);

  orbitVector.resize(elems.size());

  int i1 = 0, i2;
  for(unsigned int it = 0; it < elems.size(); it++){

    i2 = elems[it];;
    m_teapot->trackClorbit(orbit, m_ba, i1, i2);
    orbitVector[it] = orbit;

    i1 = i2;    
  }
}

void UAL::OpticsCalculator::calculateTwiss(const std::vector<int>& elems,
					   const std::vector<PacVTps>& maps,
					   std::vector<PacTwissData>& twissVector)
{
  if(!m_teapot) return;


  PAC::Position orbit;
  m_teapot->clorbit(orbit, m_ba);

  PacChromData chrom;
  m_teapot->chrom(chrom, m_ba, orbit);

  PacTwissData twiss = chrom.twiss();

  twissVector.resize(maps.size());

  double mux = 0.0, muy = 0.0;
  twiss.mu(0, mux);
  twiss.mu(1, muy);
  for(unsigned int it = 0; it < maps.size(); it++){

    m_teapot->trackTwiss(twiss, maps[it]);

    if((twiss.mu(0) - mux) < 0.0) twiss.mu(0, twiss.mu(0) + 1.0);
    mux = twiss.mu(0);

    if((twiss.mu(1) - muy) < 0.0) twiss.mu(1, twiss.mu(1) + 1.0);
    muy = twiss.mu(1);

    twissVector[it] = twiss;
  }

}

void UAL::OpticsCalculator::calculateTwiss(const std::vector<int>& elems,
					   const std::vector<PacVTps>& maps,
					   PacTwissData& tw,
					   std::vector<PacTwissData>& twissVector)
{
  if(!m_teapot) return;


  PAC::Position orbit;
  m_teapot->clorbit(orbit, m_ba);

  // PacChromData chrom;
  // m_teapot->chrom(chrom, m_ba, orbit);

  PacTwissData twiss = tw;

  twissVector.resize(maps.size());

  double mux = 0.0, muy = 0.0;
  twiss.mu(0, mux);
  twiss.mu(1, muy);
  for(unsigned int it = 0; it < maps.size(); it++){

    m_teapot->trackTwiss(twiss, maps[it]);

    if((twiss.mu(0) - mux) < 0.0) twiss.mu(0, twiss.mu(0) + 1.0);
    mux = twiss.mu(0);

    if((twiss.mu(1) - muy) < 0.0) twiss.mu(1, twiss.mu(1) + 1.0);
    muy = twiss.mu(1);

    twissVector[it] = twiss;
  }

}


void UAL::OpticsCalculator::tunefit(double tunex, double tuney,
				    std::string& b1f, std::string& b1d)
{
  if(!m_teapot) return;


  PAC::Position orbit;
  m_teapot->clorbit(orbit, m_ba);

  std::vector<int> b1fVector;
  selectElementsByNames(b1f, b1fVector);
  std::vector<int> b1dVector;
  selectElementsByNames(b1d, b1dVector);

  std::cout << "quadrupole families: " << b1fVector.size() 
	    << " " << b1dVector.size() << std::endl;

  // const PacVector<int>& b2fs;
  // const PacVector<int>& b2ds;

  m_teapot->tunethin(m_ba, orbit, b1fVector, b1dVector, tunex, tuney);

}

void UAL::OpticsCalculator::chromfit(double chromx, double chromy,
				     std::string& b2f, std::string& b2d)
{

  if(!m_teapot) return;

  PAC::Position orbit;
  m_teapot->clorbit(orbit, m_ba);

  std::vector<int> b2fVector;
  selectElementsByNames(b2f, b2fVector);
  std::vector<int> b2dVector;
  selectElementsByNames(b2d, b2dVector);

  std::cout << "sextupole families: " << b2fVector.size() 
	    << " " << b2dVector.size() << std::endl;

  // const PacVector<int>& b2fs;
  // const PacVector<int>& b2ds;

  m_teapot->chromfit(m_ba, orbit, b2fVector, b2dVector, chromx, chromy);

}

void UAL::OpticsCalculator::selectElementsByNames(const std::string& names, 
						  std::vector<int>& elemVector)
{

  if(!m_teapot) return;

  std::list<int> elemList;

  regex_t preg;
  regmatch_t pmatch[1];

  regcomp(&preg, names.c_str(), 0);   

  // std::cout << "sextupole names = " << names  << std::endl;
  for(int i = 0; i < m_teapot->size(); i++){

    TeapotElement& anode = m_teapot->element(i);
    std::string elname = anode.getDesignName();

    //  std::cout << "name = " << anode.getName()
    // << ", " << anode.getDesignName() <<std::endl;

    int rc = regexec(&preg, elname.c_str(), 1, pmatch, 0); 
    if(rc == 0) elemList.push_back(i);    
  }

  regfree(&preg); 

  elemVector.resize(elemList.size());
  int counter = 0;
  std::list<int>::iterator ie;
  for(ie = elemList.begin(); ie != elemList.end(); ie++){
    elemVector[counter++] = *ie;
  } 

}

void UAL::OpticsCalculator::writeTeapotTwissToFile(const std::string& accName,
    const std::string& fileName, const std::string& elemNames)
{

  PacLattices::iterator latIterator = PacLattices::instance()->find(accName);
  if(latIterator == PacLattices::instance()->end()){
    std::cerr << "There is no " + accName << " accelerator " << endl;
    return;
  }

  PacLattice& lattice = *latIterator; 

  std::vector<int> elems;
  selectElementsByNames(elemNames, elems);

  std::vector<double> positions;
  calculatePositions(elems, positions);

  std::vector<PacVTps> maps;
  calculateMaps(elems, maps, 1);

  std::vector<PacTwissData> twiss;
  calculateTwiss(elems, maps, twiss);

  std::ofstream out(fileName.c_str());

  std::vector<std::string> columns(11);
  columns[0]  = "#";
  columns[1]  = "name";
  columns[2]  = "suml";
  columns[3]  = "betax";
  columns[4]  = "alfax";
  columns[5]  = "qx";
  columns[6]  = "dx";
  columns[7]  = "betay";
  columns[8]  = "alfay";
  columns[9]  = "qy";
  columns[10] = "dy";

  char endLine = '\0';

  double twopi = 2.0*UAL::pi;


  out << "------------------------------------------------------------";
  out << "------------------------------------------------------------" << std::endl; 

  char line[200];
  sprintf(line, "%-5s %-10s   %-15s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s%c", 
	columns[0].c_str(),  columns[1].c_str(), columns[2].c_str(), columns[3].c_str(),  
	columns[4].c_str(),
	columns[5].c_str(), columns[6].c_str(), columns[7].c_str(), columns[8].c_str(),  
	columns[9].c_str(), columns[10].c_str(), endLine);
  out << line << std::endl;

  out << "------------------------------------------------------------";
  out << "------------------------------------------------------------" << std::endl; 

  for(int i=0; i < lattice.size(); i++){

    PacLattElement& el = lattice[i];

    sprintf(line, "%5d %-10s %15.7e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e%c", 
	    i, el.getDesignName().c_str(), positions[i], 
	    twiss[i].beta(0), twiss[i].alpha(0), 
	    twiss[i].mu(0)/twopi, twiss[i].d(0),
	    twiss[i].beta(1), twiss[i].alpha(1), 
	    twiss[i].mu(1)/twopi, twiss[i].d(1), endLine);
    out << line << std::endl;
  }

  out.close();
}

void UAL::OpticsCalculator::writeTeapotTwissToFile(const std::string& accName,
    const std::string& fileName, const std::string& elemNames, PacTwissData& tw)
{

  PacLattices::iterator latIterator = PacLattices::instance()->find(accName);
  if(latIterator == PacLattices::instance()->end()){
    std::cerr << "There is no " + accName << " accelerator " << endl;
    return;
  }

  PacLattice& lattice = *latIterator; 

  std::vector<int> elems;
  selectElementsByNames(elemNames, elems);

  std::vector<double> positions;
  calculatePositions(elems, positions);

  std::vector<PacVTps> maps;
  calculateMaps(elems, maps, 1);

  std::vector<PacTwissData> twiss;
  calculateTwiss(elems, maps, tw, twiss);

  std::ofstream out(fileName.c_str());

  std::vector<std::string> columns(11);
  columns[0]  = "#";
  columns[1]  = "name";
  columns[2]  = "suml";
  columns[3]  = "betax";
  columns[4]  = "alfax";
  columns[5]  = "qx";
  columns[6]  = "dx";
  columns[7]  = "betay";
  columns[8]  = "alfay";
  columns[9]  = "qy";
  columns[10] = "dy";

  char endLine = '\0';

  double twopi = 2.0*UAL::pi;


  out << "------------------------------------------------------------";
  out << "------------------------------------------------------------" << std::endl; 

  char line[200];
  sprintf(line, "%-5s %-10s   %-15s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s%c", 
	columns[0].c_str(),  columns[1].c_str(), columns[2].c_str(), columns[3].c_str(),  
	columns[4].c_str(),
	columns[5].c_str(), columns[6].c_str(), columns[7].c_str(), columns[8].c_str(),  
	columns[9].c_str(), columns[10].c_str(), endLine);
  out << line << std::endl;

  out << "------------------------------------------------------------";
  out << "------------------------------------------------------------" << std::endl; 

  for(int i=0; i < lattice.size(); i++){

    PacLattElement& el = lattice[i];

    sprintf(line, "%5d %-10s %15.7e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e%c", 
	    i, el.getDesignName().c_str(), positions[i], 
	    twiss[i].beta(0), twiss[i].alpha(0), 
	    twiss[i].mu(0)/twopi, twiss[i].d(0),
	    twiss[i].beta(1), twiss[i].alpha(1), 
	    twiss[i].mu(1)/twopi, twiss[i].d(1), endLine);
    out << line << std::endl;
  }

  out.close();
}







