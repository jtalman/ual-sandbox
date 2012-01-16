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

#include "positionPrinter.hh"
#include "xmgracePrint.hh"

#include "ETEAPOT/Integrator/DipoleTracker.hh"

using namespace UAL;

int main(int argc,char * argv[]){
 if(argc!=2){
 std::cout << "usage: ./evolver ./data/pre-E_pEDm.sxf (> ! myOut)\n";
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


 shell.addSplit(UAL::Args() << UAL::Arg("lattice", "ring") <<
UAL::Arg("types", "Sbend")      << UAL::Arg("ir", split));
 shell.addSplit(UAL::Args() << UAL::Arg("lattice", "ring") <<
UAL::Arg("types", "Quadrupole") << UAL::Arg("ir", 0));
 shell.addSplit(UAL::Args() << UAL::Arg("lattice", "ring") <<
UAL::Arg("types", "Sextupole")  << UAL::Arg("ir", 0));

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

#include "setBeamAttributes.hh"

 PAC::BeamAttributes& ba = shell.getBeamAttributes();

// ************************************************************************
 std::cout << "\nLinear analysis." << std::endl;
// ************************************************************************

 // Make linear matrix

 std::cout << " matrix" << std::endl;
 shell.map(UAL::Args() << UAL::Arg("order", 1) << UAL::Arg("print",
mapFile.c_str()));

 // Calculate twiss

 std::cout << " twiss (ring )" << std::endl;
 shell.twiss(UAL::Args() << UAL::Arg("print", twissFile.c_str()));

 std::cout << " calculate suml" << std::endl;
shell.analysis(UAL::Args());

// ************************************************************************
 std::cout << "\nAlgorithm Part. " << std::endl;
// ************************************************************************

 PacTMap map0(6);
 map0.mltOrder(2);

 std::string evolver_apdfFile = "./data/evolver.apdf";

 UAL::APDF_Builder apBuilder;
 apBuilder.setBeamAttributes(ba);

 UAL::AcceleratorPropagator* ap = apBuilder.parse(evolver_apdfFile);

 if(ap == 0) {
  std::cout << "Accelerator Propagator has not been created " <<
std::endl;
  return 1;
 }

 std::cout << "\n  , apdf-based evolver";  std::cout << "size : " <<
ap->getRootNode().size() << " propagators " << endl;

 UAL::PropagatorSequence& apSeq = ap->getRootNode();

 int counter = 0;
 std::list<UAL::PropagatorNodePtr>::iterator it;  for(it =
apSeq.begin(); it != apSeq.end(); it++){
  std::cout << counter++ << " " << (*it)->getType() << std::endl;  }

// ************************************************************************
 std::cout << "\nPropagate map. " << std::endl;
// ************************************************************************

 PacTMap map2(6);
 map2.setEnergy(ba.getEnergy());
 map2.mltOrder(2);

 ap->propagate(map2);

 map2.write("evolver.map");

// ************************************************************************
 std::cout << "\nElement-by-element propagation. " << std::endl;
// ************************************************************************

 counter = 0;
 char mFile[40];
 for(it = apSeq.begin(); it != apSeq.end(); it++){

 PacTMap map3(6);
 map3.setEnergy(ba.getEnergy());
 map3.mltOrder(1);

 (*it)->propagate(map3);
 std::cout << counter++ << " " << (*it)->getType() << std::endl;

 sprintf(mFile, "map%d.map", counter);
 map3.write(mFile);
 }



 return 1;
}
