#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>

#include <algorithm>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

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

//#include "positionPrinter.hh"
//#include "xmgracePrint.hh"

#include "ETEAPOT2/Integrator/bend.cc"
#include "ETEAPOT2/Integrator/quad.cc"

#include "ETEAPOT2/Integrator/genMethods/spinExtern"
#include "ETEAPOT2/Integrator/genMethods/spinDef"
#include "ETEAPOT2/Integrator/genMethods/designExtern"
#include "ETEAPOT2/Integrator/genMethods/designDef"
#include "ETEAPOT2/Integrator/genMethods/bunchParticleExtern"
#include "ETEAPOT2/Integrator/genMethods/bunchParticleDef"

using namespace UAL;

int main(int argc,char * argv[]){
 int startTime, endTime, totalTime;

#include "include/getArgs"
#include "include/setStatic"

 double f0=atof(argv[2]);

 std::string mysxf    =argv[1];
 std::string mysxfbase=mysxf.substr(7,mysxf.size()-11);
 std::cout << "mysxf     " << mysxf.c_str() << "\n";
 std::cout << "mysxfbase " << mysxfbase.c_str() << "\n";

 UAL::Shell shell;

 #include "userManifest/designBeamValues.hh"
 #include "userManifest/setBeamAttributes.hh"
 PAC::BeamAttributes& ba = shell.getBeamAttributes();

//
  mDcc=m0;
  qD=q0;

  betaD=b0;
  vD=v0*UAL::clight;
  gammaD=gamma0;

  eD=e0;
  pDc=p0;

  fD=f0;

  tofDT=0;
//tofD
//

 #include "userManifest/extractParameters.h"

 #include "userManifest/probeDataForTwiss"
 #include "userManifest/userBunch"
// #include "userManifest/simulatedProbeValues"
 int spltBndsPrBend=pow(2,splitForBends);
 //ETEAPOT2::bend::bndsPrTrn=16;   //   atoi( argv[8] )*spltBndsPrBend;
 //std::cerr << "ETEAPOT2::bend::bndsPrTrn " << ETEAPOT2::bend::bndsPrTrn << "\n";

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

 std::cout << "size : " << ap->getRootNode().size() << " propagators " << endl;

 UAL::PropagatorSequence& apSeq = ap->getRootNode();

 int counter = 0;
 std::list<UAL::PropagatorNodePtr>::iterator it;
 for(it = apSeq.begin(); it != apSeq.end(); it++){
  std::cout << counter++ << " (*it)->getType() " << (*it)->getType() << std::endl;
 }

 // ************************************************************************
 std::cout << "\nBunch Part." << std::endl;
 // ************************************************************************

 // ************************************************************************
 std::cout << "\nTracking. " << std::endl;
 // ************************************************************************

 double t; // time variable

 turns = 10;
 turns=atoi( argv[5] );
 #include"userManifest/S"
 #include"userManifest/spin"
// ETEAPOT2::mlt::initialize();
// ETEAPOT2::bend::initialize();

// positionPrinter pP;
// pP.open(orbitFile.c_str());

 ba.setElapsedTime(0.0);

 int decFac=atoi(argv[7]);
 startTime = time(NULL);
 for(int iturn = 0; iturn <= (turns-1); iturn++){
  for(int ip=0; ip < bunch.size(); ip++){
// pP.write(iturn, ip, bunch);
  }
  ap -> propagate(bunch);
 }
 endTime = time(NULL);
 totalTime = endTime - startTime;
 std::cerr << "Runtime: " << setiosflags( ios::scientific ) << setprecision(7) << totalTime << " seconds\n";

#ifdef sxfCheck
 std::cerr << "\n";
 std::cerr << "++==========================================================================================++\n";
 std::cerr << "++==========================================================================================++\n";
 std::cerr << "++==========================================================================================++\n";
 std::cerr << "++  tofDT " << tofDT << "                                                                     ++\n";
 std::cerr << "++  SXF Equilibrium Frequency: 1./tofDT " << 1./tofDT << " Design Frequency (fD): " << fD << "  ++\n";
 std::cerr << "++  thetaDT " << thetaDT << "                                                                   ++\n";
 std::cerr << "++  sDT " << sDT << "                                                                       ++\n";
 std::cerr << "++==========================================================================================++\n";
 std::cerr << "++==========================================================================================++\n";
 std::cerr << "++==========================================================================================++\n";
 std::cerr << "\n";
#endif

// pP.close();

 return 0;//1;
}
