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
#include "Main/Eteapot.h"

#include "UAL/UI/Shell.hh"

#include "PAC/Beam/Particle.hh"
#include "PAC/Beam/Spin.hh"

#include "UAL/SMF/AcceleratorNodeFinder.hh"
#include "Optics/PacTMap.h"
#include "Integrator/TeapotElemBend.h"

#include "positionPrinter.hh"
#include "xmgracePrint.hh"

#include "ETEAPOT/Integrator/DipoleTracker.hh"
#include "ETEAPOT/Integrator/MltTracker.hh"

using namespace UAL;

int main(int argc,char * argv[]){
 if(argc!=4){
  std::cout << "usage: ./manualTwiss ./data/E_Protonium.0.sxf 40 +1 (>&! OUTP1.0)\n";
  std::cout << "argv[0] is this executable         - ./manualTwiss            \n";
  std::cout << "argv[1] is the input sxf file      - ./data/E_Protonium.0.sxf \n";
  std::cout << "argv[2] is the bend radius (rinExact) of a candidate injected particle- 40\n";
  std::cout << "this does NOT override the sxf design bend radius         \n";
  std::cout << "argv[3] is the nominal electrode m - +1                   \n";
  std::cout << "                                                          \n";
  std::cout << "This radius is used to set the scale                      \n";
  std::cout << "of the probe parameters.                                  \n";
  std::cout << "It can be estimated from the sxf file(e.g.                \n";
  exit(0);
 }

  ofstream m_m;
  m_m.open ("m_m");
  m_m << argv[3];
  m_m.close();

  ETEAPOT::DipoleTracker* edt=new ETEAPOT::DipoleTracker();
  ETEAPOT::MltTracker*    mdt=new ETEAPOT::MltTracker();
  std::cerr << "ETEAPOT::DipoleTracker::m_m " << ETEAPOT::DipoleTracker::m_m << "\n";
  std::cerr << "ETEAPOT::MltTracker::m_m    " << ETEAPOT::MltTracker::m_m    << "\n";
                ETEAPOT::DipoleTracker::m_m=atof( argv[3] );
                ETEAPOT::MltTracker::m_m   =atof( argv[3] );
  std::cerr << "ETEAPOT::DipoleTracker::m_m " << ETEAPOT::DipoleTracker::m_m << "\n";
  std::cerr << "ETEAPOT::MltTracker::m_m    " << ETEAPOT::MltTracker::m_m    << "\n";

 std::string mysxf    =argv[1];
 std::string mysxfbase=mysxf.substr(7,mysxf.size()-11);
 std::cout << "mysxf     " << mysxf.c_str() << "\n";
 std::cout << "mysxfbase " << mysxfbase.c_str() << "\n";

 UAL::Shell shell;

 #include "designBeamValues.hh"
 #include "setBeamAttributes.hh"
 PAC::BeamAttributes& ba = shell.getBeamAttributes();

 #include "extractParameters.h"

 #include "probeDataForTwiss"
 #include "userBunch"
 #include "simulatedProbeValues"

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

  
 shell.addSplit(UAL::Args() << UAL::Arg("lattice", "ring") << UAL::Arg("types", "Sbend")      << UAL::Arg("ir", split-1));  // JDT 7/18/2012 new split specification
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

 // ************************************************************************
 std::cout << "\nLinear analysis." << std::endl;
 // ************************************************************************
  
 // ************************************************************************
 std::cout << "\nAlgorithm Part. " << std::endl;
 // ************************************************************************

 UAL::APDF_Builder apBuilder;

 apBuilder.setBeamAttributes(ba);

// ETEAPOT::ElectricAcceleratorPropagator* ap = (ETEAPOT::ElectricAcceleratorPropagator*) apBuilder.parse(apdfFile);
 UAL::AcceleratorPropagator* ap = apBuilder.parse(apdfFile);

 if(ap == 0) {
   std::cout << "Accelerator Propagator has not been created " << std::endl;
   return 1;
 }

 #define MAXPATHLEN 100
char cpath[MAXPATHLEN];
getcwd(cpath, MAXPATHLEN);
printf("pwd -> %s\n", cpath);
std::string path=cpath;
Eteapot* etpot;
double a0x=     (double)0;
double b0x=     (double)0;
double mu_xTent=(double)0;
double a0y=     (double)0;
double b0y=     (double)0;
double mu_yTent=(double)0;
//etpot->twissFromTracking( ba, ap, atof(argv[3]),a0x,b0x,mu_xTent,a0y,b0y,mu_yTent );
std::cerr << "RMT: a0x " << a0x << " b0x " << b0x << " mu_xTent " << mu_xTent << " a0y " << a0y << " b0y " << b0y << " mu_yTent " << mu_yTent << "\n";
//etpot->twissFromTracking( ba, ap, atof(argv[3]) );

//etpot->twissFromTracking( ba, ap, atof(argv[2]) );
//etpot->twissFromTracking( ba, ap, argv[3] );

 std::cout << "\n SXF_TRACKER manualTwiss, ";
 std::cout << "size : " << ap->getRootNode().size() << " propagators " << endl;

 UAL::PropagatorSequence& apSeq = ap->getRootNode();

 int counter = 0;
 std::list<UAL::PropagatorNodePtr>::iterator it;
 for(it = apSeq.begin(); it != apSeq.end(); it++){
  std::cout << counter++ << " (*it)->getType() " << (*it)->getType() << std::endl;
//std::cout << counter++ << " JDT - (*it)->getName() " << (*it)->getName() << " (*it)->getType() " << (*it)->getType() << std::endl;
 }

 // ************************************************************************
 std::cout << "\nBunch Part." << std::endl;
 // ************************************************************************

 // ************************************************************************
 std::cout << "\nTracking. " << std::endl;
 // ************************************************************************

 double t; // time variable

 turns = 10;

 positionPrinter pP;
 pP.open(orbitFile.c_str());
// xmgracePrint xP;
// xP.open("bunchSub0");

 ba.setElapsedTime(0.0);

 for(int iturn = 0; iturn <= (turns-1); iturn++){
//ap -> propagate(bunch);
  for(int ip=0; ip < bunch.size(); ip++){
   pP.write(iturn, ip, bunch);
// xP.write(iturn, ip, bunch);
  }
  ap -> propagate(bunch);
 }

 pP.close();
// xP.close();

 return 1;
}
