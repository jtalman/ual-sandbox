#include <iostream>
#include <fstream>

#include "UAL/APDF/APDF_Builder.hh"
#include "PAC/Beam/Position.hh"
#include "SMF/PacSmf.h"
#include "PAC/Beam/Bunch.hh"
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

  shell.readSXF(Args() << Arg("file",  "./data/muon.sxf"));

  // ************************************************************************
  std::cout << "\nAdd split ." << std::endl;
  // ************************************************************************

  shell.addSplit(Args() << Arg("lattice", "muon") << Arg("types", "Sbend")
		 << Arg("ir", 32));

  shell.addSplit(Args() << Arg("lattice", "muon") << Arg("types", "Quadrupole")
		 << Arg("ir", 32));

  // ************************************************************************
  std::cout << "Select lattice." << std::endl;
  // ************************************************************************

  shell.use(Args() << Arg("lattice", "muon"));

  // ************************************************************************
  std::cout << "\nWrite ADXF file ." << std::endl;
  // ************************************************************************

  shell.writeSXF(Args() << Arg("file",  "./out/cpp/muon.sxf"));


  // ************************************************************************
  std::cout << "\nDefine beam parameters." << std::endl;
  // ************************************************************************

  //  shell.setBeamAttributes(Args() << Arg("energy", 1.171064622)
  //			  << Arg("mass", 0.93827231));

  shell.setBeamAttributes(Args() << Arg("energy", 0.145477474)
			  << Arg("mass", 0.10565839));

  // ************************************************************************
  std::cout << "\nLinear analysis." << std::endl;
  // ************************************************************************

  // Make linear matrix
  std::cout << " matrix" << std::endl;

  shell.map(Args() << Arg("order", 1) << Arg("print", "./out/cpp/map1"));

  // Calculate twiss
  std::cout << " twiss (muon )" << std::endl;

  //  shell.twiss(Args() << Arg("print", "./out/cpp/muon.twiss"));

  // ************************************************************************
  std::cout << "2. Beam Part." << std::endl;
  // ************************************************************************

  PAC::BeamAttributes& ba = shell.getBeamAttributes();

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

  string accName = "muon";

  PacLattices::iterator latIterator = PacLattices::instance()->find(accName);
  if(latIterator == PacLattices::instance()->end()){
    std::cerr << "There is no " + accName << " accelerator " << endl;
    return 1;
  }

  PacLattice lattice = *latIterator;

  PAC::Bunch bunch(1);
  bunch.setBeamAttributes(ba);

  //  Spink tracker
  // define the intial distribution for your application

  PAC::Spin spin;
  spin.setSX(0.0);
  spin.setSY(0.0);
  spin.setSZ(1.0);

  int ip;

  for(ip=0; ip < bunch.size(); ip++){
    bunch[ip].getPosition().set(1.e-5*(ip+1), 0.0, 1.e-5*(ip+1), 0.0, 0.0, 1.e-5*(ip+1));
    bunch[ip].setSpin(spin);
  }

  std::cout << "\nSpink tracker " << endl;
  std::cout << "size : " << ap->getRootNode().size() << " propagators " << endl;

  ap->propagate(bunch);

  /*
  for(ip=0; ip < bunch.size(); ip += 1){
    PAC::Position& pos = bunch[ip].getPosition();
    std::cout << ip
  	      << " : x = " << pos.getX()
  	      << ", px = " << pos.getPX()
  	      << ", y  = " << pos.getY()
  	      << ", py = " << pos.getPY() 
	      << endl;
  }
  */
  return 1;
}

