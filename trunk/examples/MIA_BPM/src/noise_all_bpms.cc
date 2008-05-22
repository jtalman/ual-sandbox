#include "timer.h"

#include "UAL/APDF/APDF_Builder.hh"
#include "PAC/Beam/Position.hh"
#include "SMF/PacSmf.h"
#include "Main/Teapot.h"
#include "ZLIB/Tps/Space.hh"
#include "ual_sxf/Parser.hh"
#include "ACCSIM/Base/UniformGenerator.hh"

#include "BPMCollector.hh"

int main(){


  ZLIB::Space space(6, 5);

   // ************************************************************************
  std::cout << "\n1. Lattice Part." << std::endl;
  // ************************************************************************

  // string sxfFile = "blue-dAu-top-swn-no_sexts.sxf"; // "blue.sxf";
  string sxfFile = "rhic_injection.sxf";
  std::cout << "Build the Accelerator Object from the SXF file: " 
	    << sxfFile << std::endl;

  string inFile   = "./../data/"; inFile += sxfFile; 
  string echoFile = "./out/"; echoFile += sxfFile; echoFile += ".echo";
  string outFile =  "./out/"; outFile += sxfFile; outFile += ".out";

  UAL_SXF_Parser parser;
  parser.read(inFile.data(), echoFile.data()); 
  parser.write(outFile.data());

  // ************************************************************************
  std::cout << "\n2. Beam Part." << std::endl;
  // ************************************************************************

  PAC::BeamAttributes ba;
  ba.setEnergy(250.0);

  // ************************************************************************
  std::cout << "\n3. Algorithm Part. " << std::endl;
  // ************************************************************************

  std::string apdfFile = "../data/tracker.apdf";
  std::cout << "Build the Accelerator Propagator from the APDF file: " 
	    << apdfFile.data() << std::endl;

  UAL::APDF_Builder apBuilder; 
  apBuilder.setBeamAttributes(ba);
  UAL::AcceleratorPropagator* ap = apBuilder.parse(apdfFile);
  if(ap == 0) {
    std::cout << "Accelerator Propagator has not been created " << std::endl;
    return 1;
  }

  // ************************************************************************
  std::cout << "\n4. Test of APDF-based tracker. " << std::endl;
  // ************************************************************************

  string accName = "blue";

  PacLattices::iterator latIterator = PacLattices::instance()->find(accName);
  if(latIterator == PacLattices::instance()->end()){
    std::cerr << "There is no " + accName << " accelerator " << endl;
    exit(1);
  }

  PacLattice lattice = *latIterator;

  double t; // time variable
  int lsize = lattice.size();

  PAC::Bunch bunch(1); 
  bunch.setEnergy(ba.getEnergy());

  // 3.1 Teapot Tracker

  // define the intial distribution for your application

  bunch[0].getPosition().set(1.e-5*(101), 0.0, 1.e-5*(101), 0.0, 0.0, 1.e-5*(101));

  std::cout << "\nTeapot Tracker " << endl;
  std::cout << "size : " << lattice.size() << " elements " <<  endl;

  Teapot teapot(lattice);

  start_ms();
  teapot.track(bunch, 0, lsize);

  t = (end_ms());
  std::cout << "time  = " << t << " ms" << endl;

  {
    PAC::Position& pos = bunch[0].getPosition();
    std::cout << 0
	      << " : x = " << pos.getX() 
	      << ", px = " << pos.getPX() 
	      << ", y = "  << pos.getY() 
	      << ", py = " << pos.getPY() << endl;
  } 

  // 3.2 APF-based tracker 

  // define the intial distribution for your application

  bunch[0].getPosition().set(1.e-5*(101), 0.0, 1.e-5*(101), 0.0, 0.0, 1.e-5*(101));

  std::cout << "\nAPF-based Teapot tracker " << endl;  
  std::cout << "size : " << ap->getRootNode().size() << " propagators " << endl;

  start_ms();
  ap->propagate(bunch);

  t = (end_ms());
  std::cout << "time  = " << t << " ms" << endl;

  {
    PAC::Position& pos = bunch[0].getPosition();
    std::cout << 0 
	      << " : x = " << pos.getX() 
	      << ", px = " << pos.getPX() 
	      << ", y = "  << pos.getY() 
	      << ", py = " << pos.getPY() << endl;
  }   

  // ************************************************************************
  std::cout << "\n5. MIA Tracking . " << std::endl;
  // ************************************************************************

  PAC::Position clorbit;
  teapot.clorbit(clorbit, ba);

  std::cout << "closed orbit "
	    << " : x = " << clorbit.getX() 
	    << ", px = " << clorbit.getPX() 
	    << ", y = "  << clorbit.getY() 
	    << ", py = " << clorbit.getPY() << endl;

  // Define the particle position
  bunch[0].getPosition().set(clorbit.getX() + 1.e-4, 
			     clorbit.getPX(), 
			     clorbit.getY() + 1.e-4,
			     clorbit.getPY(), 
			     0.0, 
			     1.e-4);

  // Clear the previous data
  MIA::BPMCollector::getInstance().clear();

  // Propagate it over several turns 
  start_ms();

  for(int iturn = 0; iturn < 200; iturn++){
    ap->propagate(bunch);
  }

  t = (end_ms());
  std::cout << "time  = " << t << " ms" << endl;  

  // ************************************************************************
  std::cout << "\n6. Add BPM noise . " << std::endl;
  // ************************************************************************

  int iseed = -200;
  double dx = 1.0;
  double dy = 1.0;
  ACCSIM::UniformGenerator randomGenerator;

  std::map<int, MIA::BPM*>& bpms = MIA::BPMCollector::getInstance().getAllData();

  std::map<int, MIA::BPM*>::iterator ibpm;
  double r, x, y;
  for(ibpm = bpms.begin(); ibpm != bpms.end(); ibpm++){

    std::list<PAC::Position>::iterator it;
    for(it = ibpm->second->getData().begin(); it != ibpm->second->getData().end(); it++){

      x = it->getX();
      r = randomGenerator.getNumber(iseed);
      it->setX(x + dx*r);

      y = it->getY();
      r = randomGenerator.getNumber(iseed);
      it->setY(y + dy*r);
    }
  }


  // ************************************************************************
  std::cout << "\n7. Write BPM data . " << std::endl;
  // ************************************************************************

  // Write bpm turn-by-turn data into the specified file
  MIA::BPMCollector::getInstance().write("./bpm.out");

  return 1;

}
