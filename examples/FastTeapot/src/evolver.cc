#include "timer.h"

#include "UAL/APDF/APDF_Builder.hh"
#include "Optics/PacTMap.h"
#include "SMF/PacSmf.h"
#include "Main/Teapot.h"
#include "ZLIB/Tps/Space.hh"
#include "ual_sxf/Parser.hh"
#include "TEAPOT/Integrator/MapDaIntegrator.hh"
#include "UAL/APF/PropagatorFactory.hh"

int main()
{

  ZLIB::Space space(6, 5);

  // ************************************************************************
  std::cout << "1. Lattice Part." << std::endl;
  // ************************************************************************

  string sxfFile = "blue-dAu-top-swn-no_sexts.sxf";
  std::cout << " 1.2 Build the Accelerator Object from the SXF file: " 
	    << sxfFile << std::endl;

  string inFile   = "./../data/" + sxfFile; 
  string echoFile = "./out/" + sxfFile + ".echo";
  string outFile =  "./out/" + sxfFile + ".out";

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

  PacTMap map0(6);
  map0.mltOrder(2);

  std::string xmlFile = "../data/evolver.xml";

  UAL::APDF_Builder apBuilder; 
  apBuilder.setBeamAttributes(ba);

  UAL::AcceleratorPropagator* ap = apBuilder.parse(xmlFile);
  if(ap == 0) {
    std::cout << "Accelerator Propagator has not been created " << std::endl;
    return 1;
  }
  
  // ************************************************************************
  std::cout << "4. Map propagation. " << std::endl;
  // ************************************************************************

  string accName = "RHIC";

  PacLattices::iterator latIterator = PacLattices::instance()->find(accName);
  if(latIterator == PacLattices::instance()->end()){
    std::cerr << "There is no " + accName << " accelerator " << endl;
    exit(1);
  }

  PacLattice lattice = *latIterator;

  double t; // time variable
  int lsize = lattice.size(); 

  // 3.1 Teapot DA integrator

  PacTMap map1(6);
  map1.setEnergy(ba.getEnergy());
  map1.mltOrder(2);

  std::cout << "\nTeapot DA integrator " << endl;
  std::cout << "size : " << lsize << " elements " <<  endl;

  Teapot teapot(lattice);

  start_ms();
  teapot.trackMap(map1, ba, 0, lsize);
  t = (end_ms());
  std::cout << "time  = " << t << " ms" << endl;

  map1.write("teapot.map");

  // 3.2 FastTeapot DA integrator 

  PacTMap map2(6);
  map2.setEnergy(ba.getEnergy());
  map2.mltOrder(2);

  std::cout << "\nFastTeapot Da integrator " << endl;  
  std::cout << "size : " << ap->getRootNode().size() << " propagators " << endl;

  start_ms();
  ap->propagate(map2);
  t = (end_ms());
  std::cout << "time  = " << t << " ms" << endl;

  map2.write("fast_teapot.map");

  return 1;

}
