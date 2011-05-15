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
 if(argc!=4){
  std::cout << "usage: ./tracker ./data/E_FirstTest.sxf 30 -1.3 (> ! myOut)\n";
  std::cout << "argv[0] is this executable         - ./tracker\n";
  std::cout << "argv[1] is the input sxf file      - ./data/E_FirstTest.sxf\n";
  std::cout << "argv[2] is the nominal bend radius - 30      \n";
  std::cout << "argv[3] is the nominal electrode m - -1.3    \n";
  std::cout << "                                             \n";
  std::cout << "This radius is used to set the scale         \n";
  std::cout << "of the probe parameters.                     \n";
  std::cout << "It can be estimated from the sxf file(e.g.   \n";
  std::cout << "arc = 2.35619449019/                         \n";
  std::cout << "kl = 0.0785398163398 =                       \n";
  std::cout << "approximately 30).                           \n";
  std::cout << "It is a little subtle (e.g. injection issues,\n";
  std::cout << "manufacturing errors, setup errors, ...).    \n";
  std::cout << "A further subtlety is that angular           \n";
  std::cout << "momentum breaks the element-algorithm-probe  \n";
  std::cout << "paradigm, coupling probe parameter momentum  \n";
  std::cout << "with element parameter bend radius.          \n";
  exit(0);
 }

  ofstream m_m;
  m_m.open ("m_m");
  m_m << argv[3];
  m_m.close();

 std::string mysxf    =argv[1];
 std::string mysxfbase=mysxf.substr(7,mysxf.size()-11);
 std::cout << "mysxf     " << mysxf.c_str() << "\n";
 std::cout << "mysxfbase " << mysxfbase.c_str() << "\n";

 UAL::Shell shell;

 #include "designBeamValues.hh"
 #include "setBeamAttributes.hh"
 PAC::BeamAttributes& ba = shell.getBeamAttributes();

 #include "simulatedProbeValues"

 #include "extractParameters.h"

 // ************************************************************************
 std::cout << "\nDefine the space of Taylor maps." << std::endl;
 // ************************************************************************

  shell.setMapAttributes(UAL::Args() << UAL::Arg("order", order));
//shell.setMapAttributes(UAL::Args() << UAL::Arg("order", 5));

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

//#include "setBeamAttributes.hh"

// PAC::BeamAttributes& ba = shell.getBeamAttributes();

 // ************************************************************************
 std::cout << "\nLinear analysis." << std::endl;
 // ************************************************************************
  
/*
 // Make linear matrix

 std::cout << " matrix" << std::endl;
 shell.map(UAL::Args() << UAL::Arg("order", 1) << UAL::Arg("print", mapFile.c_str()));

 // Calculate twiss
  
 std::cout << " twiss (ring )" << std::endl;
 shell.twiss(UAL::Args() << UAL::Arg("print", twissFile.c_str()));

 std::cout << " calculate suml" << std::endl;
 shell.analysis(UAL::Args());
*/

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

/*
 PAC::Bunch bunch(1);               // bunch with 1 particle(s)
 bunch.setBeamAttributes(ba);

 PAC::Spin spin;
 spin.setSX(0.0);
 spin.setSY(0.0);
 spin.setSZ(1.0);

  bunch[0].getPosition().set(dx,dpx,dy,dpy,0,p5Input);
*/
/*
  bunch[0].getPosition().set(1.e-4,0.    ,1.e-4,0.    ,0.,0.);
  bunch[1].getPosition().set(0.   ,0.5e-5,0.   ,0.    ,0.,0.);
  bunch[2].getPosition().set(0.   ,0.    ,1.e-4,0.    ,0.,0.);
  bunch[3].getPosition().set(0.   ,0.    ,0.   ,0.5e-6,0.,0.);
*/

 // ************************************************************************
 std::cout << "\nTracking. " << std::endl;
 // ************************************************************************

 double t; // time variable

// int turns = 1024;

 positionPrinter pP;
 pP.open(orbitFile.c_str());
 xmgracePrint xP;
 xP.open("bunchSub0");

 ba.setElapsedTime(0.0);

 for(int iturn = 1; iturn <= turns; iturn++){
  ap -> propagate(bunch);
  for(int ip=0; ip < bunch.size(); ip++){
   pP.write(iturn, ip, bunch);
   xP.write(iturn, ip, bunch);
  }
 }

 pP.close();
 xP.close();

 return 1;
}
