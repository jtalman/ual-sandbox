#include <iostream>
#include <fstream>
#include <iomanip>

#include <stdio.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "UAL/APDF/APDF_Builder.hh"
#include "PAC/Beam/Position.hh"
#include "SMF/PacSmf.h"
#include "PAC/Beam/Bunch.hh"
#include "Main/Teapot.h"
#include "UAL/UI/Shell.hh"

#include "PAC/Beam/Particle.hh"
#include "PAC/Beam/Spin.hh"

#include "UAL/SMF/AcceleratorNodeFinder.hh"

#include "TEAPOT/Integrator/DipoleData.hh"
#include "TEAPOT/Integrator/MagnetData.hh"

#include "timer.h"
#include "positionPrinter.hh"
#include "threeVector.hh"
#include "fourVector.hh"
#include "fourTensor.hh"
#include "lorentzTransformForTracker.cc"

#include "globalBlock.cc"

using namespace UAL;

int main(int argc,char * argv[]){

  // Input parameters

  // ************************************************************************
  std::cout << "\nEcho input parameters." << std::endl;
  // ************************************************************************

  if(argc < 3) {
    std::cout << "Usage : ./tracker <InputFile, e.g. muon0.13_R5m-mod> <BendSplit, e.g. 1> <QuadSplit, e.g. 4>" << std::endl;
    std::cout << "All units are M.K.S. except ee which is in GeV" << std::endl;
    exit(1);
  }

  std::string variantName = argv[1];    // std::string sxfInputFile   = "../sxf/muon0.13_R5m.sxf";
  int bendsplit = atof(argv[2]);        // std::integer bendslit = 1;
  int quadsplit = atof(argv[3]);        // std::integer quadsplit = 4;

  UAL::Shell shell;

  // ************************************************************************
  std::cout << "\nDefine the space of Taylor maps." << std::endl;
  // ************************************************************************

  shell.setMapAttributes(UAL::Args() << UAL::Arg("order", 5));

  // ************************************************************************
  std::cout << "\nBuild lattice." << std::endl;
  // ************************************************************************

  std::string sxfFile = "./sxf/";
  sxfFile += variantName;
  sxfFile += ".sxf";

  shell.readSXF(UAL::Args() << UAL::Arg("file",  sxfFile.c_str()));

  // ************************************************************************
  std::cout << "\nAdd split ." << std::endl;
  // ************************************************************************

  
  shell.addSplit(UAL::Args() << UAL::Arg("lattice", "ring") << UAL::Arg("types", "Sbend")
		 << UAL::Arg("ir", bendsplit));

  shell.addSplit(UAL::Args() << UAL::Arg("lattice", "ring") << UAL::Arg("types", "Quadrupole")
		 << UAL::Arg("ir", quadsplit));
  

  // ************************************************************************
  std::cout << "Select lattice." << std::endl;
  // ************************************************************************

  shell.use(UAL::Args() << UAL::Arg("lattice", "ring"));

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

  double mass   = 0.10578404; // muon rest mass
  m0 = 0.10578404;
  p0 = .1;
  e0 = sqrt(m0*m0 + p0*p0);
  c0 = 1.60193E-19;
  shell.setBeamAttributes(UAL::Args() << UAL::Arg("energy", e0) << UAL::Arg("mass", m0) << UAL::Arg("charge", c0));
  PAC::BeamAttributes& ba = shell.getBeamAttributes();
  ba.setG(0.0011659230);             // adds muon G factor
  G0 = ba.getG();

  std::cout << "m0 = " << m0 << "\n";
  std::cout << "p0 = " << p0 << "\n";
  std::cout << "e0 = " << e0 << "\n";
  std::cout << "c0 = " << c0 << "\n";
  std::cout << "G0 = " << G0 << "\n";

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

  std::cout << " twiss (ring )" << std::endl;

  shell.twiss(UAL::Args() << UAL::Arg("print", twissFile.c_str()));

  std::cout << " calculate suml" << std::endl;
  shell.analysis(UAL::Args());

  // ************************************************************************
  std::cout << "\nAlgorithm Part. " << std::endl;
  // ************************************************************************

//std::string apdfFile = argv[1];
  std::string apdfFile = "./apdf/thinspin.apdf";

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

  PAC::Bunch bunch(1);               // bunch with one particle
  bunch.setBeamAttributes(ba);

//gsl_rng *r = gsl_rng_alloc(gsl_rng_default);
  gsl_rng *r;
  const gsl_rng_type * T;
  gsl_rng_env_setup();
  T = gsl_rng_default;
  r = gsl_rng_alloc(T);
  std::cout << "generator type: " << gsl_rng_name(r) << "\n";
  std::cout << "seed: " << gsl_rng_default_seed << "\n";
  std::cout << "first value: " << gsl_rng_get(r) << "\n";
  float theta, phi;

  for(int ip=0; ip < bunch.size(); ip ++){
       bunch[ip].getPosition().set(0.0, 0.0, 0.0, 1.0E-3, 0.0, 0.0);
    // bunch[ip].getPosition().set(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    const PAC::Position pos = bunch[ip].getPosition();
#include "set_betal.cc"
std::cout << "gamma " << gamma << "\n";
std::cout << "beta " << beta << "\n";
    theta=gsl_ran_flat(r,0,M_PI);
    phi  =gsl_ran_flat(r,0,2*M_PI);
std::cout << "theta " << theta << "\n";
std::cout << "phi " << phi << "\n";
    theta=gsl_ran_flat(r,0,M_PI);
    phi  =gsl_ran_flat(r,0,2*M_PI);
std::cout << "theta " << theta << "\n";
std::cout << "phi " << phi << "\n";
    spin.setSX(sin(theta)*cos(phi));
    spin.setSY(sin(theta)*sin(phi));
    spin.setSZ(cos(theta));
 
    bunch[ip].setSpin(spin);

    SR[ip].set0(0);
    SR[ip].set1(spin.getSX());
    SR[ip].set2(spin.getSY());
    SR[ip].set3(spin.getSZ());

//  SR[ip].set(sr0,sr1,sr2,sr3);
 
    lorentzTransformForTracker(SR[ip], betal*=(-1), gamma, SL[ip]);
std::cout << "SR: SR0 " << SR[ip].get0() << " SR1 " << SR[ip].get1() << " SR2 " << SR[ip].get2() << " SR3 " << SR[ip].get3() << "\n";
std::cout << "SL: SR0 " << SL[ip].get0() << " SL1 " << SL[ip].get1() << " SL2 " << SL[ip].get2() << " SL3 " << SL[ip].get3() << "\n";
  }

/*
  sr.set(sr0,sr1,sr2,sr3);           // sr0 == 0 !!!
  const PAC::Position pos = bunch[0].getPosition();
#include "set_betal.cc"
std::cout << "gamma " << gamma << "\n";
std::cout << "beta " << beta << "\n";
  lorentzTransformForTracker(sr, betal*=(-1), gamma, sl);
  sl0 = sl.get0();
  sl1 = sl.get1();
  sl2 = sl.get2();
  sl3 = sl.get3();
std::cout << "sl: sl0 " << sl0 << " sl1 " << sl1 << " sl2 " << sl2 << " sl3 " << sl3 << "\n";
//sl.set(spin0,spin1,spin2,spin3);
*/

  ul.set(0,0,0,0);
  pl.set(0,0,0,0);

 // ************************************************************************
  std::cout << "\nTracking. " << std::endl;
  // ************************************************************************

  double t; // time variable

  int turns = 10;

  std::string orbitFile = "./out/cpp/";
  orbitFile += variantName;
  orbitFile += ".orbit";

  positionPrinter pP;
  pP.open(orbitFile.c_str());

  ba.setElapsedTime(0.0);

  start_ms();

  for(int iturn = 1; iturn <= turns; iturn++){
    std::cout << "turn loop: turn # " << iturn << "\n";

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

