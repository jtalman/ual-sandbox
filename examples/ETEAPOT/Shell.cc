
#include <ctype.h>
#include <regex.h> 

#include "UAL/Common/Def.hh"
#include "UAL/APDF/APDF_Builder.hh"
#include "UAL/SMF/AcceleratorNodeFinder.hh"

#include "Optics/PacChromData.h"
#include "UAL/SXF/Parser.hh"
#include "SIMBAD/SC/TSCPropagatorFFT.hh"
#include "SIMBAD/SC/LSCCalculatorFFT.hh"
#include "SIMBAD/SC/TSCCalculatorFFT.hh"

#include "Optics/PacTMap.h"

#include "UAL/ADXF/Reader.hh"
#include "UAL/ADXF/Writer.hh"

#include "UAL/UI/Shell.hh"
#include "UAL/UI/ShellImp.hh"

UAL::Shell::Shell()
{
  m_ap = 0;
}

bool UAL::Shell::setMapAttributes(const UAL::Arguments& arguments)
{
  std::map<std::string, UAL::Argument*>::const_iterator it;

  const std::map<std::string, UAL::Argument*>& args = arguments.getMap();

  // "order"

  int order = 1;

  it = args.find("order");
  if(it != args.end()){
    order = (int) it->second->getNumber();
  }
  UAL::ShellImp::getInstance().m_space = new ZLIB::Space(6, order);
  
  return true;
}

bool UAL::Shell::setBeamAttributes(const UAL::Arguments& arguments)
{
  std::map<std::string, UAL::Argument*>::const_iterator it;

  const std::map<std::string, UAL::Argument*>& args = arguments.getMap();

  it = args.find("energy");                         // e0 - 1
  if(it != args.end()){
    m_ba.setEnergy(it->second->getNumber());
  }

  it = args.find("mass");                           // m0 - 2
  if(it != args.end()){
    m_ba.setMass(it->second->getNumber());
  }

  it = args.find("charge");                         // q0 - 3
  if(it != args.end()){
    m_ba.setCharge(it->second->getNumber());
  }

  it = args.find("macrosize");                      // M0 - 6
  if(it != args.end()){
    m_ba.setMacrosize(it->second->getNumber());
  }

  it = args.find("designAngularMomentum");          // L0 - 8 (5th chronologically)
  if(it != args.end()){
    m_ba.setL(it->second->getNumber());
  }

  it = args.find("designElectricField");            // E0 - 9 (ordering?)
  if(it != args.end()){
    m_ba.setE(it->second->getNumber());
  }

  it = args.find("designRadius");                   // R0 - 10 (ordering?)
  if(it != args.end()){
    m_ba.setR(it->second->getNumber());
  }

  it = args.find("frequency");                      // 
  if(it != args.end()){
    m_ba.setRevfreq(it->second->getNumber());
  }

  it = args.find("gyromagnetic");                   // 
  if(it != args.end()){
    m_ba.setG(it->second->getNumber());
  }

  it = args.find("gFactor");                        // 
  if(it != args.end()){
    m_ba.set_g(it->second->getNumber());
  }

  UAL::OpticsCalculator::getInstance().setBeamAttributes(m_ba);
  
  return true;
}

PAC::BeamAttributes& UAL::Shell::getBeamAttributes()
{
  return m_ba;
}

bool UAL::Shell::setBunch(const UAL::Arguments& arguments)
{
  return m_bunchGenerator.setBunchArguments(arguments);
}

void UAL::Shell::updateBunch()
{
  m_bunch.setBeamAttributes(m_ba);
  m_bunchGenerator.updateBunch(m_bunch,
			       UAL::OpticsCalculator::getInstance().m_chrom->twiss());
}

PAC::Bunch& UAL::Shell::getBunch()
{
  return m_bunch;
}

bool UAL::Shell::readADXF(const UAL::Arguments& arguments)
{
  std::map<std::string, UAL::Argument*>::const_iterator it;

  const std::map<std::string, UAL::Argument*>& args = arguments.getMap();
  
  // "file"

  std::string adxfFile;

  it = args.find("file");
  if(it == args.end()){
    std::cerr << "readADXF:: sxf file is not defined " << std::endl;
    return false;
  }
  
  adxfFile = it->second->getString();

  // Initialize the ADXF reader
  UAL::ADXFReader* reader = UAL::ADXFReader::getInstance();
  if(reader == 0) {
    std::cout << "reader == 0 " << std::endl;
    return 0;
  }

  // Read data
  reader->read(adxfFile.c_str());

  // "print"
  
  std::string echoFile = "./echo.sxf";

  it = args.find("print");
  if(it != args.end()){
    echoFile = it->second->getString();
  } else {
    return true;
  }

  std::cout << "echo file " << echoFile << std::endl;

  // Write echo
  UAL::SXFParser writer;
  writer.write(echoFile.c_str());

  return true;
}

bool UAL::Shell::writeADXF(const UAL::Arguments& arguments)
{
  std::map<std::string, UAL::Argument*>::const_iterator it;
  const std::map<std::string, UAL::Argument*>& args = arguments.getMap();
  
  // "file"

  std::string adxfFile;

  it = args.find("file");
  if(it == args.end()){
    std::cerr << "writeADXF: adxf file is not defined " << std::endl;
    return false;
  }
  
  adxfFile = it->second->getString();

  // Write data
  
  UAL::ADXFWriter adxfWriter;
  adxfWriter.write(adxfFile.c_str());

  return true;
}



bool UAL::Shell::readSXF(const UAL::Arguments& arguments)
{
  std::map<std::string, UAL::Argument*>::const_iterator it;

  const std::map<std::string, UAL::Argument*>& args = arguments.getMap();
  
  // "file"

  std::string sxfFile;

  it = args.find("file");
  if(it == args.end()){
    std::cerr << "readSXF:: sxf file is not defined " << std::endl;
    return false;
  }
  
  sxfFile = it->second->getString();
  
  // "print"
  
  std::string echoFile = "./echo.sxf";

  it = args.find("print");
  if(it != args.end()){
    echoFile = it->second->getString();
  }

  std::cout << "echo file " << echoFile << std::endl;

  // read file

  UAL::SXFParser parser;
  parser.read(sxfFile.c_str(), echoFile.data()); 

  return true;
}

bool UAL::Shell::writeSXF(const UAL::Arguments& arguments)
{
  std::map<std::string, UAL::Argument*>::const_iterator it;

  const std::map<std::string, UAL::Argument*>& args = arguments.getMap();
  
  // "file"

  std::string sxfFile;

  it = args.find("file");
  if(it == args.end()){
    std::cerr << "writeSXF:: sxf file is not defined " << std::endl;
    return false;
  }
 
  sxfFile = it->second->getString();
  // std::cout << "file = " << sxfFile << std::endl;

  // write file

  UAL::SXFParser parser;
  parser.write(sxfFile.c_str()); 

  return true;
}

bool UAL::Shell::use(const UAL::Arguments& arguments)
{
  std::map<std::string, UAL::Argument*>::const_iterator it;

  const std::map<std::string, UAL::Argument*>& args = arguments.getMap();
  
  // "lattice"

  it = args.find("lattice");
  if(it == args.end()){
    std::cerr << "use:: lattice name is not defined " << std::endl;
    return false;
  }

  std::string accName = it->second->getString();

  PacLattices::iterator latIterator = PacLattices::instance()->find(accName);
  if(latIterator == PacLattices::instance()->end()){
    std::cerr << "There is no " + accName << " accelerator " << endl;
    return false;
  }

  m_accName = accName;
  // UAL::OpticsCalculator::getInstance().m_teapot->use(*latIterator);
  UAL::OpticsCalculator::getInstance().use(accName);

  return true;
}

bool UAL::Shell::analysis(const UAL::Arguments& arguments)
{
  std::map<std::string, UAL::Argument*>::const_iterator it;

  const std::map<std::string, UAL::Argument*>& args = arguments.getMap();
  
  // "print"

  std::string fileName = "./analysis";

  it = args.find("print");
  if(it != args.end()){
    fileName = it->second->getString();
  }

  UAL::OpticsCalculator& optics = UAL::OpticsCalculator::getInstance();

  // update lattice
  optics.calculate();

  std::cout << "suml = " << optics.suml << std::endl;

  it = args.find("twiss");

  PacTwissData* twissPtr = 0;
  if(it != args.end()){
    twissPtr = &static_cast<PacTwissData&>( it->second->getObject() );
  }

  if(twissPtr) *twissPtr = optics.m_chrom->twiss();

  return true;
  
}

bool UAL::Shell::tunefit(const UAL::Arguments& arguments)
{
  std::map<std::string, UAL::Argument*>::const_iterator it;

  const std::map<std::string, UAL::Argument*>& args = arguments.getMap();

  UAL::OpticsCalculator& optics = UAL::OpticsCalculator::getInstance();

  // update lattice
  optics.calculate();

  // "tunesx"

  double tunex =  optics.m_chrom->twiss().mu(0)/2./UAL::pi;

  it = args.find("tunex");
  if(it != args.end()){
    tunex = it->second->getNumber();
  }

  double tuney =  optics.m_chrom->twiss().mu(1)/2./UAL::pi;

  it = args.find("tuney");
  if(it != args.end()){
    tuney = it->second->getNumber();
  }

  std::string b1f, b1d;

  it = args.find("b1f");
  if(it != args.end()){
    b1f = it->second->getString();
  } else {
    std::cout << "b1f is not defined" << std::endl;
    return false;
  }

  it = args.find("b1d");
  if(it != args.end()){
    b1d = it->second->getString();
  } else {
    std::cout << "b1d is not defined" << std::endl;
    return false;    
  }

  optics.tunefit(tunex, tuney, b1f, b1d);
  optics.calculate();

  return true;
  
}

bool UAL::Shell::chromfit(const UAL::Arguments& arguments)
{
  std::map<std::string, UAL::Argument*>::const_iterator it;

  const std::map<std::string, UAL::Argument*>& args = arguments.getMap();

  UAL::OpticsCalculator& optics = UAL::OpticsCalculator::getInstance();

  // update lattice
  optics.calculate();

  // "chromx"

  double chromx =  optics.m_chrom->dmu(0)/2./UAL::pi;

  it = args.find("chromx");
  if(it != args.end()){
    chromx = it->second->getNumber();
  }

  double chromy =  optics.m_chrom->dmu(1)/2./UAL::pi;

  it = args.find("chromy");
  if(it != args.end()){
    chromy = it->second->getNumber();
  }

  std::string b2f, b2d;

  it = args.find("b2f");
  if(it != args.end()){
    b2f = it->second->getString();
  } else {
    std::cout << "b2f is not defined" << std::endl;
    return false;
  }

  it = args.find("b2d");
  if(it != args.end()){
    b2d = it->second->getString();
  } else {
    std::cout << "b2d is not defined" << std::endl;
    return false;    
  }

  optics.chromfit(chromx, chromy, b2f, b2d);
  optics.calculate();

  return true;
  
}

bool UAL::Shell::map(const UAL::Arguments& arguments)
{
  std::map<std::string, UAL::Argument*>::const_iterator it;

  const std::map<std::string, UAL::Argument*>& args = arguments.getMap();
  
  // "print"

  std::string fileName = "./map";

  it = args.find("print");
  if(it != args.end()){
    fileName = it->second->getString();
  }

  // "order"

  int  order = 1;

  it = args.find("order");
  if(it != args.end()){
    order = (int) it->second->getNumber();
  }

  UAL::OpticsCalculator& optics = UAL::OpticsCalculator::getInstance();

  PAC::Position orbit;
  optics.m_teapot->clorbit(orbit, m_ba);

  PacTMap map(6);
  map.refOrbit(orbit);
  optics.m_teapot->map(map, m_ba, order);

  map.write(fileName.c_str());

  return true;
}



bool UAL::Shell::readAPDF(const UAL::Arguments& arguments)
{
  std::map<std::string, UAL::Argument*>::const_iterator it;

  const std::map<std::string, UAL::Argument*>& args = arguments.getMap();
  
  // "file"

  it = args.find("file");
  if(it == args.end()){
    std::cerr << "readAPDF:: apdf file is not defined " << std::endl;
    return false;
  }

  m_apdfFile = it->second->getString();

  // read file
  
  UAL::APDF_Builder apBuilder; 
  apBuilder.setBeamAttributes(m_ba);

  if(m_ap != 0) delete m_ap;
  m_ap = apBuilder.parse(m_apdfFile);
  if(m_ap == 0) {
    std::cout << "Accelerator Propagator has not been created " << std::endl;
    return false;
  }  

  return true;
}

bool UAL::Shell::rebuildPropagator()
{
  if(m_ap != 0) delete m_ap;

  UAL::APDF_Builder apBuilder; 
  apBuilder.setBeamAttributes(m_ba);

  m_ap = apBuilder.parse(m_apdfFile);
  if(m_ap == 0) {
    std::cout << "Accelerator Propagator has not been created " << std::endl;
    return false;
  }  
  return true;
}

bool UAL::Shell::run(const UAL::Arguments& arguments)
{
  std::map<std::string, UAL::Argument*>::const_iterator it;

  const std::map<std::string, UAL::Argument*>& args = arguments.getMap();
  
  // "turns"

  int turns = 1;

  it = args.find("turns");
  if(it != args.end()){
    turns = (int) it->second->getNumber();
  }

  PAC::Bunch* bunchPtr = 0;

  it = args.find("bunch");
  if(it != args.end()){
    bunchPtr = &static_cast<PAC::Bunch&>( it->second->getObject() );
  }

  if(bunchPtr == 0) return false;

  // char line[120];
    
  for(int it = 0; it < turns; it++){

    UAL::PropagatorSequence& seq = m_ap->getRootNode();
    UAL::PropagatorIterator ip;

    for(ip = seq.begin(); ip != seq.end(); ip++){
      (*ip)->propagate(*bunchPtr);
    }
  }

  return true;

}

void UAL::Shell::addSplit(const UAL::Arguments& arguments)
{
  std::map<std::string, UAL::Argument*>::const_iterator it;
  const std::map<std::string, UAL::Argument*>& args = arguments.getMap();

  // "lattice"

  it = args.find("lattice");
  if(it == args.end()){
    std::cerr << "use:: lattice name is not defined " << std::endl;
    return;
  }

  std::string accName = it->second->getString();

  // "ir"
  int ir = 1;

  it = args.find("ir");
  if(it != args.end()){
    ir = (int) it->second->getNumber();
  }

  std::string elementTypes;

  it = args.find("types");
  if(it != args.end()){
    elementTypes = it->second->getString();
  }

  std::vector<int> elems;
  selectElementsByTypes(accName, elementTypes, elems);

  UAL::AcceleratorNode* lattice = getLattice(accName); 

  for(unsigned int ie = 0; ie < elems.size(); ie++){
    PacLattElement* anode = 
      static_cast<PacLattElement*>( lattice->getNodeAt(elems[ie]));
    anode->addN(ir);
  }

}

void UAL::Shell::getMaps(const UAL::Arguments& arguments, 
			 std::vector<PacVTps>& maps)
{
  std::map<std::string, UAL::Argument*>::const_iterator it;
  const std::map<std::string, UAL::Argument*>& args = arguments.getMap();

  // "order"

  int order = 1;

  it = args.find("order");
  if(it != args.end()){
    order = (int) it->second->getNumber();
  }

  // elementTypes

  std::string elementTypes;

  it = args.find("types");
  if(it != args.end()){
    elementTypes = it->second->getString();
  }

  std::vector<int> elems;
  selectElementsByTypes(m_accName, elementTypes, elems);

  UAL::OpticsCalculator& optics = UAL::OpticsCalculator::getInstance();

  maps.clear();
  optics.calculateMaps(elems, maps, order);

}



void UAL::Shell::getTwiss(const UAL::Arguments& arguments, 
			  std::vector<double>& positions,
			  std::vector<PacTwissData>& twiss)
{
  std::map<std::string, UAL::Argument*>::const_iterator it;
  const std::map<std::string, UAL::Argument*>& args = arguments.getMap();

  // elementTypes

  std::string elementTypes;

  it = args.find("types");
  if(it != args.end()){
    elementTypes = it->second->getString();
  }

  std::vector<int> elems;
  selectElementsByTypes(m_accName, elementTypes, elems);

  UAL::OpticsCalculator& optics = UAL::OpticsCalculator::getInstance();

  optics.calculatePositions(elems, positions);

  std::vector<PacVTps> maps;
  optics.calculateMaps(elems, maps, 1);

  twiss.clear();
  optics.calculateTwiss(elems, maps, twiss);

}

bool UAL::Shell::twiss(const UAL::Arguments& arguments, PacTwissData& tw)
{
std::cout << "File " << __FILE__ << " line " << __LINE__ << " method bool UAL::Shell::twiss(const UAL::Arguments& arguments, PacTwissData& tw)\n";
  std::map<std::string, UAL::Argument*>::const_iterator it;
  const std::map<std::string, UAL::Argument*>& args = arguments.getMap(); 

  // "print"

  std::string fileName = "./twiss";

  it = args.find("print");
  if(it != args.end()){
    fileName = it->second->getString();
  }
  
  
  // elements

  std::string elemNames;

  it = args.find("elements");
  if(it != args.end()){
    elemNames= it->second->getString();
  }

  UAL::OpticsCalculator& optics = UAL::OpticsCalculator::getInstance();
  
  optics.writeTeapotTwissToFile(m_accName, fileName, elemNames, tw);
  
  return true;
}

bool UAL::Shell::twiss(const UAL::Arguments& arguments)
{
std::cout << "File " << __FILE__ << " line " << __LINE__ << " enter method bool UAL::Shell::twiss(const UAL::Arguments& arguments)\n";
  std::map<std::string, UAL::Argument*>::const_iterator it;
  const std::map<std::string, UAL::Argument*>& args = arguments.getMap(); 

  // "print"

  std::string fileName = "./twiss";

  it = args.find("print");
  if(it != args.end()){
    fileName = it->second->getString();
  }
  
  
  // elements

  std::string elemNames;

  it = args.find("elements");
  if(it != args.end()){
    elemNames= it->second->getString();
  }

  UAL::OpticsCalculator& optics = UAL::OpticsCalculator::getInstance();
  
std::cout << "File " << __FILE__ << " line " << __LINE__ << " before optics.writeTeapotTwissToFile(m_accName, fileName, elemNames);\n";
  optics.writeTeapotTwissToFile(m_accName, fileName, elemNames);
std::cout << "File " << __FILE__ << " line " << __LINE__ << " after  optics.writeTeapotTwissToFile(m_accName, fileName, elemNames);\n";
  
std::cout << "File " << __FILE__ << " line " << __LINE__ << " leave method bool UAL::Shell::twiss(const UAL::Arguments& arguments)\n";
  return true;
}



UAL::AcceleratorNode* UAL::Shell::getLattice(const std::string& accName)
{
  UAL::AcceleratorNodeFinder::Iterator it = 
    UAL::AcceleratorNodeFinder::getInstance().find(accName);

  UAL::AcceleratorNode* lattice = 0;

  if(it != UAL::AcceleratorNodeFinder::getInstance().end()){
    lattice = (it->second).operator->();
  }

  return lattice;
}

void UAL::Shell::selectElementsByTypes(const std::string& accName, 
				       const std::string& types, 
				       std::vector<int>& elemVector)
{

  UAL::AcceleratorNode* lattice = getLattice(accName); 

  std::list<int> elemList;

  regex_t preg;
  regmatch_t pmatch[1];

  regcomp(&preg, types.c_str(), 0);   

  for(int i = 0; i < lattice->getNodeCount(); i++){

    UAL::AcceleratorNode* const anode = lattice->getNodeAt(i);
    std::string eltype = anode->getType();

    int rc = regexec(&preg, eltype.c_str(), 1, pmatch, 0); 
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



