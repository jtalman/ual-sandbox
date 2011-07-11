#include <iostream>
#include <fstream>
#include <iomanip>

#include <stdio.h>

#include "UAL/APDF/APDF_Builder.hh"
#include "PAC/Beam/Position.hh"
#include "SMF/PacSmf.h"
#include "PAC/Beam/Bunch.hh"
#include "Main/Teapot.h"
#include "UAL/UI/Shell.hh"

#include "PAC/Beam/Particle.hh"
#include "PAC/Beam/Spin.hh"

#include "UAL/SMF/AcceleratorNodeFinder.hh"

#include "timer.h"
#include "positionPrinter.hh"

using namespace UAL;

int main(int argc,char * argv[]){

  UAL::Shell shell;

  // std::string variantName = "muon_R5m-RFon";
  std::string variantName = "muon0.13_R5m";

  // ************************************************************************
  std::cout << "\nDefine the space of Taylor maps." << std::endl;
  // ************************************************************************

  shell.setMapAttributes(UAL::Args() << UAL::Arg("order", 5));

  // ************************************************************************
  std::cout << "\nBuild lattice." << std::endl;
  // ************************************************************************

  std::string sxfFile = "./data/";
  sxfFile += variantName;
  sxfFile += ".sxf";

  shell.readSXF(UAL::Args() << UAL::Arg("file",  sxfFile.c_str()));

  // ************************************************************************
  std::cout << "\nAdd split ." << std::endl;
  // ************************************************************************

  
  shell.addSplit(UAL::Args() << UAL::Arg("lattice", "muon") << UAL::Arg("types", "Sbend")
		 << UAL::Arg("ir", 4));

  shell.addSplit(UAL::Args() << UAL::Arg("lattice", "muon") << UAL::Arg("types", "Quadrupole")
		 << UAL::Arg("ir", 4));
  

  // ************************************************************************
  std::cout << "Select lattice." << std::endl;
  // ************************************************************************

  shell.use(UAL::Args() << UAL::Arg("lattice", "muon"));

  // ************************************************************************
  std::cout << "\nWrite SXF file ." << std::endl;
  // ************************************************************************

  std::string outputFile = "./out/cpp/";
  outputFile += variantName;
  outputFile += ".sxf";

  shell.writeSXF(UAL::Args() << UAL::Arg("file",  outputFile.c_str()));

  // ************************************************************************
  std::cout << "\nDefine beam parameters." << std::endl;
  // ************************************************************************

  double mass   = 0.10565839; // muon rest mass
  double energy = sqrt(mass*mass + 0.1*0.1);

  shell.setBeamAttributes(UAL::Args() << UAL::Arg("mass", mass));
  shell.setBeamAttributes(UAL::Args() << UAL::Arg("energy", energy));
//shell.setBeamAttributes(UAL::Args() << UAL::Arg("energy", energy) << UAL::Arg("mass", mass));

  PAC::BeamAttributes& ba = shell.getBeamAttributes();

  // ************************************************************************
  std::cout << "\nLinear analysis." << std::endl;
  // ************************************************************************
  
  // Make linear matrix

  std::string mapFile = "./out/cpp/";
  mapFile += variantName;
  mapFile += ".map1";

  std::cout << " matrix" << std::endl;
  shell.map(UAL::Args() << UAL::Arg("order", 1) << UAL::Arg("print", mapFile.c_str()));

  // Calculate twiss
  
  std::string twissFile = "./out/cpp/";
  twissFile += variantName;
  twissFile += ".twiss";

  std::cout << " twiss (muon )" << std::endl;

  shell.twiss(UAL::Args() << UAL::Arg("print", twissFile.c_str()));

  std::cout << " calculate suml" << std::endl;
  shell.analysis(UAL::Args());

  // ************************************************************************
  std::cout << "\nAlgorithm Part. " << std::endl;
  // ************************************************************************

  std::string apdfFile = argv[1];

  UAL::APDF_Builder apBuilder;

  apBuilder.setBeamAttributes(ba);

  UAL::AcceleratorPropagator* ap = apBuilder.parse(apdfFile);

  if(ap == 0) {
    std::cout << "Accelerator Propagator has not been created " << std::endl;
    return 1;
  }

  std::cout << "\n SXF_TRACKER tracker, ";
  std::cout << "size : " << ap->getRootNode().size() << " propagators " << endl;

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

//int turns = 3000;
  int turns = 10;

  std::string orbitFile = "./out/cpp/";
  orbitFile += variantName;
  orbitFile += ".orbit";

  positionPrinter pP;
  pP.open(orbitFile.c_str());

  ba.setElapsedTime(0.0);

  start_ms();

  for(int iturn = 1; iturn <= turns; iturn++){

    ap -> propagate(bunch);

    for(int ip=0; ip < bunch.size(); ip++){
       pP.write(iturn, ip, bunch);
    }
  }

  t = (end_ms());
  std::cout << "time  = " << t << " ms" << endl;

  pP.close();

  return 1;
}

