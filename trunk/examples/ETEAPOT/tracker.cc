#include <iostream>
#include <fstream>
#include <iomanip>

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

//#include "timer.h"
//#include "positionPrinter.hh"

using namespace UAL;

int main(int argc,char * argv[]){
int pause;
std::cin >> pause;
std::cout << "file  $UAL/examples/SXF_TRACKER/" << __FILE__ << " line " << __LINE__ << " method int main(int argc,char * argv[])" << " about to " << "UAL::Shell shell;" << "\n";

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
  sxfFile += argv[2];
//sxfFile += variantName;
  sxfFile += ".sxf";
std::cout << "\nJDT -- sxfFile " << sxfFile << std::endl;

  shell.readSXF(UAL::Args() << UAL::Arg("file",  sxfFile.c_str()));

  // ************************************************************************
  std::cout << "\nAdd split ." << std::endl;
  // ************************************************************************

  
  shell.addSplit(UAL::Args() << UAL::Arg("lattice", "ring") << UAL::Arg("types", "Sbend")      << UAL::Arg("ir", 4));
  shell.addSplit(UAL::Args() << UAL::Arg("lattice", "ring") << UAL::Arg("types", "Quadrupole") << UAL::Arg("ir", 2));
  shell.addSplit(UAL::Args() << UAL::Arg("lattice", "ring") << UAL::Arg("types", "Sextupole")  << UAL::Arg("ir", 2));
 /* 
  shell.addSplit(UAL::Args() << UAL::Arg("lattice", "muon") << UAL::Arg("types", "Sbend")      << UAL::Arg("ir", 1));
  shell.addSplit(UAL::Args() << UAL::Arg("lattice", "muon") << UAL::Arg("types", "Quadrupole") << UAL::Arg("ir", 2));
  shell.addSplit(UAL::Args() << UAL::Arg("lattice", "muon") << UAL::Arg("types", "Sextupole")  << UAL::Arg("ir", 2));
*/

  // ************************************************************************
  std::cout << "Select lattice." << std::endl;
  // ************************************************************************

  shell.use(UAL::Args() << UAL::Arg("lattice", "ring"));
//shell.use(UAL::Args() << UAL::Arg("lattice", "muon"));

  // ************************************************************************
  std::cout << "\nWrite SXF file ." << std::endl;
  // ************************************************************************

  std::string outputFile = "./out/cpp/";
  outputFile += argv[2];
//outputFile += variantName;
  outputFile += ".sxf";
std::cout << "\nJDT -- outputFile " << outputFile << std::endl;

  shell.writeSXF(UAL::Args() << UAL::Arg("file",  outputFile.c_str()));

  // ************************************************************************
  std::cout << "\nDefine beam parameters." << std::endl;
  // ************************************************************************

//double mass   = 0.10565839; //   muon rest mass
  double mass   = 0.93827231; // proton rest mass
  double m0=mass;
  double chge   = 1.6e-19   ; // proton charge
  double gamma0 = atof(argv[3]);
  double v0= UAL::clight*sqrt(1-1/gamma0/gamma0);
  double energy = gamma0*m0;
  double e0=energy;
  double p0 = gamma0*m0*v0;
//double energy = sqrt(mass*mass + p*p);
//double energy = sqrt(mass*mass + 0.70074*0.70074);
//double energy = sqrt(mass*mass + 0.1*0.1);
  std::cout << "\nEnergy " << energy << std::endl;

  shell.setBeamAttributes(UAL::Args() << UAL::Arg("energy", energy) << UAL::Arg("mass", mass));
//shell.setBeamAttributes(UAL::Args() << UAL::Arg("elapsedTime", 0));

  PAC::BeamAttributes& ba = shell.getBeamAttributes();

  // ************************************************************************
  std::cout << "\nLinear analysis." << std::endl;
  // ************************************************************************
  
  // Make linear matrix

  std::string mapFile = "./out/cpp/";
  mapFile += argv[2];
//mapFile += variantName;
  mapFile += ".map1";

  std::cout << " matrix" << std::endl;
  shell.map(UAL::Args() << UAL::Arg("order", 1) << UAL::Arg("print", mapFile.c_str()));

  // Calculate twiss
  
  std::string twissFile = "./out/cpp/";
  twissFile += argv[2];
//twissFile += variantName;
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

double R0   = atof(argv[5]);
double E0   = atof(argv[6]);
double dx   = 0;            double x = R0 + dx;
double gpx  = 0;
double dy   = 0;            double y = dy;
double gpy  = atof(argv[4]);
double cdt  = 0;
double dz   = cdt;          double z = dz;
double gpz  = 1;                               // ??? !!!
double r    = sqrt(x*x+y*y+z*z);
double p    = sqrt(gpx*gpx+gpy*gpy+gpz*gpz);
double Escr = sqrt(p*p+m0*m0)+chge*R0*log(r/R0);;

PAC::Position posWork;
posWork.set(dx,gpx,dy,gpy,cdt,(Escr-energy)/(p0*UAL::clight));

  for(int ip=0; ip < bunch.size(); ip ++){
    bunch[ip].getPosition().set(0.0, 0.0, 0.0,gpy   , 0.0, 0.0);
//  bunch[ip].getPosition().set(0.0, 0.0, 0.0, 1.0E-3, 0.0, 0.0);
    // bunch[ip].getPosition().set(0.0, 0.0, 0.0, 0.0E-3, 0.0, 0.0);
    bunch[ip].setSpin(spin);
  }

 // ************************************************************************
  std::cout << "\nTracking. " << std::endl;
  // ************************************************************************

  double t; // time variable

//int turns = 3000;
//int turns = 10;
//int turns = 2;
  int turns = 1;

  std::string orbitFile = "./out/cpp/";
  orbitFile += argv[2];
//orbitFile += variantName;
  orbitFile += ".orbit";

//positionPrinter pP;
//pP.open(orbitFile.c_str());

  ba.setElapsedTime(0.0);

//start_ms();

  for(int iturn = 1; iturn <= turns; iturn++){

    ap -> propagate(bunch);

    for(int ip=0; ip < bunch.size(); ip++){
//     pP.write(iturn, ip, bunch);
    }
  }

//t = (end_ms());
//std::cout << "time  = " << t << " ms" << endl;

//pP.close();

  return 1;
}

