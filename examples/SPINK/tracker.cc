

#include "UAL/APDF/APDF_Builder.hh"
#include "PAC/Beam/Position.hh"
#include "SMF/PacSmf.h"
#include "Main/Teapot.h"
#include "UAL/UI/Shell.hh"

using namespace UAL;

int main(){

  UAL::Shell shell;

  // ************************************************************************
  std::cout << "\nDefine the space of Taylor maps." << std::endl;
  // ************************************************************************

  shell.setMapAttributes(Args() << Arg("order", 5));


  // ************************************************************************
  std::cout << "\nBuild lattice." << std::endl;
  // ************************************************************************

  shell.readSXF(Args() << Arg("file",  "./data/proton_ring.sxf"));

  // ************************************************************************
  std::cout << "\nAdd split ." << std::endl;
  // ************************************************************************

  shell.addSplit(Args() << Arg("lattice", "edm") << Arg("types", "Sbend")
		 << Arg("ir", 2));

  shell.addSplit(Args() << Arg("lattice", "edm") << Arg("types", "Quadrupole")
		 << Arg("ir", 2));

  // ************************************************************************
  std::cout << "Select lattice." << std::endl;
  // ************************************************************************

  shell.use(Args() << Arg("lattice", "edm"));

  // ************************************************************************
  std::cout << "\nWrite ADXF file ." << std::endl;
  // ************************************************************************

  shell.writeSXF(Args() << Arg("file",  "./out/cpp/proton_ring.sxf"));


  // ************************************************************************
  std::cout << "\nDefine beam parameters." << std::endl;
  // ************************************************************************

  shell.setBeamAttributes(Args() << Arg("energy", 1.93827231)
			  << Arg("mass", 0.93827231));

  // ************************************************************************
  std::cout << "\nLinear analysis." << std::endl;
  // ************************************************************************

  // Make linear matrix
  std::cout << " matrix" << std::endl;

  shell.map(Args() << Arg("order", 1) << Arg("print", "./out/cpp/map1"));

  // Calculate twiss
  std::cout << " twiss (edm )" << std::endl;

  shell.twiss(Args() << Arg("print", "./out/cpp/edm.twiss"));

  // ************************************************************************
  std::cout << "2. Beam Part." << std::endl;
  // ************************************************************************

  PAC::BeamAttributes ba;
  ba.setEnergy(250.0);

  // ************************************************************************
  std::cout << "3. Algorithm Part. " << std::endl;
  // ************************************************************************

  std::string xmlFile = "data/spink.apdf";

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

  string accName = "edm";

  PacLattices::iterator latIterator = PacLattices::instance()->find(accName);
  if(latIterator == PacLattices::instance()->end()){
    std::cerr << "There is no " + accName << " accelerator " << endl;
    return 1;
  }

  PacLattice lattice = *latIterator;

  // double t; // time variable
  int lsize = lattice.size();

  PAC::Bunch bunch(100);
  bunch.setBeamAttributes(ba);

  // 3.1 Teapot Tracker

  // define the intial distribution for your application

  int ip;
  for(ip=0; ip < bunch.size(); ip++){
    bunch[ip].getPosition().set(1.e-5*(ip+1), 0.0, 1.e-5*(ip+1), 0.0, 0.0, 1.e-5*(ip+1));
  }

  std::cout << "\nTeapot Tracker " << endl;
  std::cout << "size : " << lattice.size() << " elements " <<  endl;

  Teapot teapot(lattice);

  // start_ms();
  teapot.track(bunch, 0, lsize);
  // t = (end_ms());
  // std::cout << "time  = " << t << " ms" << endl;

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

  // start_ms();

  // int counter = 0;
  ap->propagate(bunch);

  // t = (end_ms());
  // std::cout << "time  = " << t << " ms" << endl;

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

