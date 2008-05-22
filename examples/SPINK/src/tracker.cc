#include "timer.h"

#include "UAL/APDF/APDF_Builder.hh"
#include "PAC/Beam/Position.hh"
#include "SMF/PacSmf.h"
#include "Main/Teapot.h"
#include "ZLIB/Tps/Space.hh"
#include "ual_sxf/Parser.hh"

int main(){

  ZLIB::Space space(6, 5);

   // ************************************************************************
  std::cout << "1. Lattice Part." << std::endl;
  // ************************************************************************

  string sxfFile = "rhic.sxf"; // "blue-dAu-top-swn-no_sexts.sxf"; // "blue.sxf";
  std::cout << "1.2 Build the Accelerator Object from the SXF file: " 
	    << sxfFile << std::endl;

  string inFile   = "./../data/"; inFile  += sxfFile; 
  string echoFile = "./out/"; echoFile += sxfFile; echoFile += ".echo";
  string outFile =  "./out/"; outFile += sxfFile; outFile += ".out";

  UAL_SXF_Parser parser;
  parser.read(inFile.data(), echoFile.data()); 
  parser.write(outFile.data());

  // ************************************************************************
  std::cout << "2. Beam Part." << std::endl;
  // ************************************************************************

  PAC::BeamAttributes ba;
  ba.setEnergy(250.0);

  // ************************************************************************
  std::cout << "3. Algorithm Part. " << std::endl;
  // ************************************************************************

  std::string xmlFile = "../data/tracker.apdf";

  UAL::APDF_Builder apBuilder; 
  apBuilder.setBeamAttributes(ba);
  UAL::AcceleratorPropagator* ap = apBuilder.parse(xmlFile);
  if(ap == 0) {
    std::cout << "Accelerator Propagator has not been created " << std::endl;
    return 1;
  }

  // ************************************************************************
  std::cout << "4. Tracking. " << std::endl;
  // ************************************************************************

  string accName = "yellow";

  PacLattices::iterator latIterator = PacLattices::instance()->find(accName);
  if(latIterator == PacLattices::instance()->end()){
    std::cerr << "There is no " + accName << " accelerator " << endl;
    exit(1);
  }

  PacLattice lattice = *latIterator;

  double t; // time variable
  int lsize = lattice.size();

  PAC::Bunch bunch(100); 
  bunch.setEnergy(ba.getEnergy());

  // 3.1 Teapot Tracker

  // define the intial distribution for your application

  int ip;
  for(ip=0; ip < bunch.size(); ip++){
    bunch[ip].getPosition().set(1.e-5*(ip+1), 0.0, 1.e-5*(ip+1), 0.0, 0.0, 1.e-5*(ip+1));
  }

  std::cout << "\nTeapot Tracker " << endl;
  std::cout << "size : " << lattice.size() << " elements " <<  endl;

  Teapot teapot(lattice);

  start_ms();
  teapot.track(bunch, 0, lsize);
  t = (end_ms());
  std::cout << "time  = " << t << " ms" << endl;

  for(ip=0; ip < bunch.size(); ip += 10){
    PAC::Position& pos = bunch[ip].getPosition();
    std::cout << ip 
	      << " : x = " << pos.getX() 
	      << ", px = " << pos.getPX() 
	      << ", y = "  << pos.getY() 
	      << ", py = " << pos.getPY() << endl;
  } 

  // 3.2 Spink tracker 

  // define the intial distribution for your application

  for(ip=0; ip < bunch.size(); ip ++){
    bunch[ip].getPosition().set(1.e-5*(ip+1), 0.0, 1.e-5*(ip+1), 0.0, 0.0, 1.e-5*(ip+1));
  }

  std::cout << "\nSpink tracker " << endl;  
  std::cout << "size : " << ap->getRootNode().size() << " propagators " << endl;

  start_ms();
  int counter = 0;
  ap->propagate(bunch);
  t = (end_ms());
  std::cout << "time  = " << t << " ms" << endl;

  for(ip=0; ip < bunch.size(); ip += 10){
    PAC::Position& pos = bunch[ip].getPosition();
    std::cout << ip 
	      << " : x = " << pos.getX() 
	      << ", px = " << pos.getPX() 
	      << ", y = "  << pos.getY() 
	      << ", py = " << pos.getPY() << endl;
  }   

  return 1;

}
