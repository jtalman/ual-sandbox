
#include "UAL/Common/Def.hh"
#include "UAL/APDF/APDF_Builder.hh"
#include "UAL/SXF/Parser.hh"
#include "SIMBAD/SC/TSCPropagatorFFT.hh"
#include "SIMBAD/SC/LSCCalculatorFFT.hh"
#include "SIMBAD/SC/TSCCalculatorFFT.hh"

#include "Shell.hh"

UAL::Shell::Shell()
{
  m_space = 0;
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
  m_space = new ZLIB::Space(6, order);
  
  return true;
}

bool UAL::Shell::setBeamAttributes(const UAL::Arguments& arguments)
{
  std::map<std::string, UAL::Argument*>::const_iterator it;

  const std::map<std::string, UAL::Argument*>& args = arguments.getMap();

  it = args.find("energy");
  if(it != args.end()){
    m_ba.setEnergy(it->second->getNumber());
  }

  it = args.find("macrosize");
  if(it != args.end()){
    m_ba.setMacrosize(it->second->getNumber());
  }
  
  return true;
}



PAC::BeamAttributes& UAL::Shell::getBeamAttributes()
{
  return m_ba;
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

  // read file

  UAL::SXFParser parser;
  parser.read(sxfFile.c_str(), echoFile.data()); 

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

  m_lattice = *latIterator;
  m_teapot.use(m_lattice);

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

  PacSurveyData surveyData;
  m_teapot.survey(surveyData);

  double suml = surveyData.survey().suml();
  std::cout << "suml = " << suml << std::endl;

  PAC::Position orbit;
  m_teapot.clorbit(orbit, m_ba);

  PacChromData chrom;
  m_teapot.chrom(chrom, m_ba, orbit);

  it = args.find("twiss");

  PacTwissData* twissPtr = 0;
  if(it != args.end()){
    twissPtr = &static_cast<PacTwissData&>( it->second->getObject() );
  }

  if(twissPtr) *twissPtr = chrom.twiss();

  return true;
  
}

bool UAL::Shell::readAPDF(const UAL::Arguments& arguments)
{
  std::map<std::string, UAL::Argument*>::const_iterator it;

  const std::map<std::string, UAL::Argument*>& args = arguments.getMap();
  
  // "file"

  std::string apdfFile;

  it = args.find("file");
  if(it == args.end()){
    std::cerr << "readAPDF:: apdf file is not defined " << std::endl;
    return false;
  }

  apdfFile = it->second->getString();

  // read file
  
  UAL::APDF_Builder apBuilder; 
  apBuilder.setBeamAttributes(m_ba);

  m_ap = apBuilder.parse(apdfFile);
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

  SIMBAD::LSCCalculatorFFT& lscFFT =  SIMBAD::LSCCalculatorFFT::getInstance(); 

  // char line[120];
    
  for(int it = 0; it < turns; it++){

    UAL::PropagatorSequence& seq = m_ap->getRootNode();
    UAL::PropagatorIterator ip;
    int counter = 0;

    // lscFFT.defineLFactors(*bunchPtr);

    for(ip = seq.begin(); ip != seq.end(); ip++){

      // SIMBAD::TSCPropagatorFFT* scp = 
      // 	static_cast<SIMBAD::TSCPropagatorFFT*> ((*ip).getPointer());

      (*ip)->propagate(*bunchPtr);

      // if(counter == 0) {
      // 	SIMBAD::TSCCalculatorFFT::getInstance().showForce("forceEnd.out");
      // 	exit(1);
      // }


      /*
      std::cout << counter << " " 
		<< m_lattice[counter].getDesignName() << " " 
		<< m_lattice[counter].getName() 
		<< " l = " << m_lattice[counter].getLength() << std::endl;
      PAC::Position& p = bunch[0].getPosition();
      sprintf (line, "x=%14.8e px=%14.8e y=%14.8e py=%14.8e ct=%14.8e dE/p=%14.8e",
	       p.getX(), p.getPX(), p.getY(), p.getPY(), p.getCT(), p.getDE()); 
      std::cout << line << std::endl;
      */
      counter++;
    }

    // SIMBAD::TSCCalculatorFFT::getInstance().showForce("forceEnd.out");

  }

  return true;

}



