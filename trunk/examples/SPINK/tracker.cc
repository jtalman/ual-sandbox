#include <iostream>
#include <fstream>
#include <iomanip>

#include "UAL/APDF/APDF_Builder.hh"
#include "PAC/Beam/Position.hh"
#include "SMF/PacSmf.h"
#include "PAC/Beam/Bunch.hh"
#include "Main/Teapot.h"
#include "UAL/UI/Shell.hh"

#include "SPINK/Propagator/DipoleErTracker.hh"
#include "SPINK/Propagator/SpinTrackerWriter.hh"

#include "timer.h"
#include "PositionPrinter.h"
#include "SpinPrinter.h"

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

  // shell.readSXF(Args() << Arg("file",  "./data/muon_R5m.sxf"));
  shell.readSXF(Args() << Arg("file",  "./data/muon0.13_R5m.sxf"));

  // ************************************************************************
  std::cout << "\nAdd split ." << std::endl;
  // ************************************************************************

  shell.addSplit(Args() << Arg("lattice", "muon") << Arg("types", "Sbend")
		 << Arg("ir", 4));

  shell.addSplit(Args() << Arg("lattice", "muon") << Arg("types", "Quadrupole")
		 << Arg("ir", 4));

  // ************************************************************************
  std::cout << "Select lattice." << std::endl;
  // ************************************************************************

  shell.use(Args() << Arg("lattice", "muon"));

  // ************************************************************************
  std::cout << "\nWrite ADXF file ." << std::endl;
  // ************************************************************************

  shell.writeSXF(Args() << Arg("file",  "./out/cpp/muon0.13_R5m.sxf"));

  // ************************************************************************
  std::cout << "\nDefine beam parameters." << std::endl;
  // ************************************************************************


  double mass   = 0.10565839; // muon rest mass
  double energy = sqrt(mass*mass + 0.1*0.1);

  shell.setBeamAttributes(Args() << Arg("energy", energy) << Arg("mass", mass));

  PAC::BeamAttributes& ba = shell.getBeamAttributes();

  // ************************************************************************
  std::cout << "\nLinear analysis." << std::endl;
  // ************************************************************************

  // Make linear matrix
  std::cout << " matrix" << std::endl;
  shell.map(Args() << Arg("order", 1) << Arg("print", "./out/cpp/map1"));

  // Calculate twiss
  std::cout << " twiss (muon )" << std::endl;
  shell.twiss(Args() << Arg("print", "./out/cpp/muon.twiss"));

  std::cout << " calculate suml" << std::endl;
  shell.analysis(Args());

  // ************************************************************************
  std::cout << "\nAlgorithm Part. " << std::endl;
  // ************************************************************************

  std::string apdfFile = "data/spink.apdf";

  UAL::APDF_Builder apBuilder;

  apBuilder.setBeamAttributes(ba);

  UAL::AcceleratorPropagator* ap = apBuilder.parse(apdfFile);

  if(ap == 0) {
    std::cout << "Accelerator Propagator has not been created " << std::endl;
    return 1;
  }

  std::cout << "\n Spink tracker, ";
  std::cout << "size : " << ap->getRootNode().size() << " propagators " << endl;

  // ************************************************************************
  std::cout << "\nElectric field. " << std::endl;
  // ************************************************************************

  //  double ER  = 0.012; // GV/m
  double ER  = 0.0, EV = 0.0, EL = 0.0; // GV/m  
  SPINK::DipoleErTracker::setER(ER);
  SPINK::DipoleErTracker::setEV(EV);
  SPINK::DipoleErTracker::setEL(EL);

  // ************************************************************************
  std::cout << "\nBunch Part." << std::endl;
  // ************************************************************************

  ba.setG(0.0011659230);             // adds muon G factor

  PAC::Bunch bunch(1);               // bunch with one particle
  bunch.setBeamAttributes(ba);

  PAC::Spin spin;
  spin.setSX(0.0);
  spin.setSY(0.0);
  spin.setSZ(1.0);

  for(int ip=0; ip < bunch.size(); ip ++){
    bunch[ip].getPosition().set(0.0, 0.0, 0.0, 1.0E-3, 0.0, 0.0);
    // bunch[ip].getPosition().set(0.0, 0.0, 0.0, 0.0E-3, 0.0, 0.0);
    bunch[ip].setSpin(spin);
  }

 // ************************************************************************
  std::cout << "\nTracking. " << std::endl;
  // ************************************************************************

  double t; // time variable

  int turns = 1000000;

  SPINK::SpinTrackerWriter* stw = SPINK::SpinTrackerWriter::getInstance(); 
  stw->setFileName("spin_finestep.dat");

  PositionPrinter positionPrinter;
  positionPrinter.open("orbit.dat");
  
  SpinPrinter spinPrinter;
  spinPrinter.open("spin.dat");

  ba.setElapsedTime(0.0);

  for(int iturn = 1; iturn <= turns; iturn++){   
    for(int ip=0; ip < bunch.size(); ip++){

      ap -> propagate(bunch);

      positionPrinter.write(iturn, ip, bunch);
      spinPrinter.write(iturn, ip, bunch);

    }
  }

  positionPrinter.close();
  spinPrinter.close();

  return 1;
}

