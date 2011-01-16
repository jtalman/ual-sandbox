#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>

#include <algorithm>

#include <stdio.h>
#include <stdlib.h>

#include "UAL/APDF/APDF_Builder.hh"
#include "PAC/Beam/Position.hh"
#include "SMF/PacSmf.h"
#include "PAC/Beam/Bunch.hh"
#include "Main/Teapot.h"
#include "UAL/UI/Shell.hh"

#include "PAC/Beam/Particle.hh"
#include "PAC/Beam/Spin.hh"

#include "UAL/SMF/AcceleratorNodeFinder.hh"
#include "Optics/PacTMap.h"
#include "Integrator/TeapotElemBend.h"

//#include "timer.h"
#include "positionPrinter.hh"

//#include "globalBlock.h"

#include "ETEAPOT/Integrator/DipoleTracker.hh"

using namespace UAL;

int main(int argc,char * argv[]){
 if(argc!=2){
  std::cout << "usage: ./tracker ./data/pre-E_pEDm.sxf (> ! myOut)\n";
  std::cout << "argv[0] is this executable: ./tracker\n";
  std::cout << "argv[1] is the input sxf file - ./data/pre-E_pEDm.sxf\n";
  exit(0);
 }

std::string mysxf    =argv[1];
std::string mysxfbase=mysxf.substr(7,mysxf.size()-11);
std::cout << "mysxf     " << mysxf.c_str() << "\n";
std::cout << "mysxfbase " << mysxfbase.c_str() << "\n";

#include "designBeamValues.hh"
#include "extractParameters.h"

 UAL::Shell shell;

 // ************************************************************************
 std::cout << "\nDefine the space of Taylor maps." << std::endl;
 // ************************************************************************

 shell.setMapAttributes(UAL::Args() << UAL::Arg("order", 5));

 // ************************************************************************
 std::cout << "\nBuild lattice." << std::endl;
 // ************************************************************************

 shell.readSXF(UAL::Args() << UAL::Arg("file",  sxfFile.c_str()));

 // ************************************************************************
 std::cout << "\nAdd split ." << std::endl;
 // ************************************************************************

  
 shell.addSplit(UAL::Args() << UAL::Arg("lattice", "ring") << UAL::Arg("types", "Sbend")      << UAL::Arg("ir", split));
 shell.addSplit(UAL::Args() << UAL::Arg("lattice", "ring") << UAL::Arg("types", "Quadrupole") << UAL::Arg("ir", 0));
 shell.addSplit(UAL::Args() << UAL::Arg("lattice", "ring") << UAL::Arg("types", "Sextupole")  << UAL::Arg("ir", 0));

 // ************************************************************************
 std::cout << "Select lattice." << std::endl;
 // ************************************************************************

 shell.use(UAL::Args() << UAL::Arg("lattice", "ring"));

 // ************************************************************************
 std::cout << "\nWrite SXF file ." << std::endl;
 // ************************************************************************

 shell.writeSXF(UAL::Args() << UAL::Arg("file",  outputFile.c_str()));

 // ************************************************************************
 std::cout << "\nDefine beam parameters." << std::endl;
 // ************************************************************************

// shell.setBeamAttributes(UAL::Args() << UAL::Arg("energy", e0) << UAL::Arg("mass", m0));
// shell.setBeamAttributes(UAL::Args() << UAL::Arg("elapsedTime", 0));
#include "setBeamAttributes.hh"

 PAC::BeamAttributes& ba = shell.getBeamAttributes();

 // ************************************************************************
 std::cout << "\nLinear analysis." << std::endl;
 // ************************************************************************
  
 // Make linear matrix

 std::cout << " matrix" << std::endl;
 shell.map(UAL::Args() << UAL::Arg("order", 1) << UAL::Arg("print", mapFile.c_str()));

 // Calculate twiss
  
 std::cout << " twiss (ring )" << std::endl;
 shell.twiss(UAL::Args() << UAL::Arg("print", twissFile.c_str()));

 std::cout << " calculate suml" << std::endl;
 shell.analysis(UAL::Args());

 // ************************************************************************
 std::cout << "\nAlgorithm Part. " << std::endl;
 // ************************************************************************

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

// ba.setG(1.7928474);             // adds proton G factor

 PAC::Bunch bunch(1);               // bunch with one particle
 bunch.setBeamAttributes(ba);

 PAC::Spin spin;
 spin.setSX(0.0);
 spin.setSY(0.0);
 spin.setSZ(1.0);

 std::cout << "probeEscr0 " << probeEscr0 << "\n";

 for(int ip=0; ip < bunch.size(); ip ++){
  bunch[ip].getPosition().set(probe__dx0,probe_dpx0,probe__dy0,probe_dpy0,probe_cdt0,probeEscr0);
  bunch[ip].setSpin(spin);
 }

 // ************************************************************************
 std::cout << "\nTracking. " << std::endl;
 // ************************************************************************

 double t; // time variable

 int turns = 1;

 positionPrinter pP;
 pP.open(orbitFile.c_str());

 ba.setElapsedTime(0.0);

 for(int iturn = 1; iturn <= turns; iturn++){
  ap -> propagate(bunch);
  for(int ip=0; ip < bunch.size(); ip++){
   pP.write(iturn, ip, bunch);
  }
 }
 pP.close();
 return 1;
}
