#include <iostream>
#include <fstream>
#include <iomanip>

#include "UAL/APDF/APDF_Builder.hh"
#include "PAC/Beam/Position.hh"
#include "SMF/PacSmf.h"
#include "PAC/Beam/Bunch.hh"
#include "Main/Teapot.h"
#include "UAL/UI/Shell.hh"

#include "SPINK/Propagator/DipoleTracker.hh"
#include "SPINK/Propagator/RFCavityTracker.hh"
#include "SPINK/Propagator/SpinTrackerWriter.hh"

#include "timer.h"
#include "PositionPrinter.h"
#include "SpinPrinter.h"

using namespace UAL;

int main(){

  UAL::Shell shell;

  std::ifstream configInput("./data/spink.in");//AULNLD:07JAN10

  //std::string variantName = "blue_split_run06_snakes_polarstart";
  std::string variantName;
  configInput >> variantName;

  //  std::string variantName = "blue_split_run06_snakes";

  // ************************************************************************
  std::cout << "\nDefine the space of Taylor maps." << std::endl;
  // ************************************************************************

   shell.setMapAttributes(Args() << Arg("order", 5));

  // ************************************************************************
  std::cout << "\nBuild lattice." << std::endl;
  // ************************************************************************

  std::string sxfFile = "./data/";
  sxfFile += variantName;
  sxfFile += ".sxf";

  std::cout << "sxfFile = " << sxfFile << endl;

  shell.readSXF(Args() << Arg("file",  sxfFile.c_str()));

  // ************************************************************************
  std::cout << "\nAdd split ." << std::endl;
  // ************************************************************************
  int irSBend;
  int irQuad;
  configInput >> irSBend >> irQuad;
  std::cout << "irSBend = " << irSBend << ", irQuad = " << irQuad << endl;

  shell.addSplit(Args() << Arg("lattice", "ring") << Arg("types", "Sbend")
  		 << Arg("ir", irSBend));

  shell.addSplit(Args() << Arg("lattice", "ring") << Arg("types", "Quadrupole")
  		 << Arg("ir", irQuad));

  // ************************************************************************
  std::cout << "Select lattice." << std::endl;
  // ************************************************************************

  shell.use(Args() << Arg("lattice", "ring"));

  // ************************************************************************
  std::cout << "\nWrite ADXF file ." << std::endl;
  // ************************************************************************

  std::string outputFile = "./out/teapot/";
  outputFile += variantName;
  outputFile += ".sxf";

  shell.writeSXF(Args() << Arg("file",  outputFile.c_str()));

  // ************************************************************************
  std::cout << "\nDefine beam parameters." << std::endl;
  // ************************************************************************

  double cc  = 2.99792458E+8;
  double G = 1.7928456;

  //double gamma = 205. ; // 143.; 25.379;   266.336;  145.4;  266.609; 
  double gamma;
  configInput >> gamma;

  double mass   = 0.938272029;            //       proton mass [GeV]
  double energy = gamma*mass;
  double charge = 1.0;

  shell.setBeamAttributes(Args() << Arg("energy", energy) << Arg("mass", mass)
			  << Arg("charge",charge));

  PAC::BeamAttributes& ba = shell.getBeamAttributes();

  // ************************************************************************
  std::cout << "\nLinear analysis." << std::endl;
  // ************************************************************************
  
  // Make linear matrix

  std::string mapFile = "./out/teapot/";
  mapFile += variantName;
  mapFile += ".map1";

  std::cout << " matrix" << std::endl;
  shell.map(Args() << Arg("order", 1) << Arg("print", mapFile.c_str()));

  // ************************************************************************
  std::cout << "\nTune and chromaticity fitting. " << std::endl;
  // ************************************************************************

  // shell.analysis(Args());
  double tuneX, tuneY, chromX, chromY;
  configInput >> tuneX >> tuneY >> chromX >> chromY;
  //  std::cout << "tuneX = " << tuneX << ", tuneY = " << tuneY 
  //	    << ", chromX = " << chromX << ", chromY = " << chromY << endl;
  //    shell.tunefit(Args() << Arg("tunex", 28.69) << Arg("tuney", 29.69) << Arg("b1f", "^qf$") << Arg("b1d", "^qd$"));
  shell.tunefit(Args() << Arg("tunex", tuneX) << Arg("tuney", tuneY) << Arg("b1f", "^qf$") << Arg("b1d", "^qd$"));
  shell.chromfit(Args() << Arg("chromx", chromX) << Arg("chromy", chromY)<< Arg("b2f", "^sf$") << Arg("b2d", "^sd$"));

  // Calculate twiss
  
  std::string twissFile = "./out/teapot/";
  twissFile += variantName;
  twissFile += ".twiss";

  std::cout << " twiss " << std::endl;

  
  shell.twiss(Args() << Arg("print", twissFile.c_str()));

  std::cout << " calculate suml" << std::endl;
  shell.analysis(Args());

  // ************************************************************************
  std::cout << "\nAlgorithm Part. " << std::endl;
  // ************************************************************************

  std::string apdfFile = "./data/teapot.apdf"; // nikolay 1/15/10 replacing spink.apdf with teapot.apdf

  UAL::APDF_Builder apBuilder;

  apBuilder.setBeamAttributes(ba);

  UAL::AcceleratorPropagator* ap = apBuilder.parse(apdfFile);

  if(ap == 0) {
    std::cout << "Accelerator Propagator has not been created " << std::endl;
    return 1;
  }

  std::cout << "\nSpink tracker, ";
  std::cout << "size : " << ap->getRootNode().size() << " propagators " << endl;

  // ************************************************************************
  std::cout << "\nSet Acceleration. " << std::endl;
 // ************************************************************************

  //double dgammadt = 1.1522754; // 1/s
  double dgammadt;
  configInput >> dgammadt;
  double dedt = dgammadt*mass;
  double circum = OpticsCalculator::getInstance().suml; 
  double T_0 = circum / cc;

  //double V = 1.5e-04;
  double V;
  configInput >> V;
  double lag = asin((dedt * T_0)/(2*V))/(2*UAL::pi);

  // double V = 0.;
  // double lag = 0.;

  cout << "dgamma/dt = " << dgammadt << endl ; //AUL:29DEC09
  cout << "Circumference(m) = " << circum << endl;
    
  SPINK::RFCavityTracker::setRF(V, 360, lag);  

 
  // ************************************************************************
  std::cout << "\nBunch Part." << std::endl;
  // ************************************************************************

  ba.setG(G);         // proton G factor
  
  cout << "gamma = " << gamma << ",  Ggamma = " << G*gamma << endl;

  PAC::Bunch bunch(1);               // bunch with one particle
  bunch.setBeamAttributes(ba);

  PAC::Spin spin;
  spin.setSX(0.0);
  spin.setSY(1.0);
  spin.setSZ(0.0);

  //double amplit_y = 15.; // Pi mm*mrad (normalized) 15.; 
  //double amplit_x = 0.; // Pi mm*mrad (normalized)
  //double dpp0 = 0.0;
  
  double amplit_y; // Pi mm*mrad (normalized) 15.; 
  double amplit_x; // Pi mm*mrad (normalized)
  double dpp0;

  configInput >> amplit_x >> amplit_y >> dpp0;


  cout << "emit_x = " << amplit_x << ", emit_y = " << amplit_y << ", dpp0= " << dpp0 <<  endl; //AUL:30DEC09

  // ************************************************************************
  std::cout << "\nOptics" << std::endl; //AUL:30DEC09
  // ************************************************************************

  UAL::OpticsCalculator& optics = UAL::OpticsCalculator::getInstance();

  optics.calculate();

  PacTwissData tws = optics.m_chrom->twiss();
  double q_x = tws.mu(0)/2./UAL::pi;
  double q_y = tws.mu(1)/2./UAL::pi;
  double beta_x = tws.beta(0);
  double beta_y = tws.beta(1);
  double chrm_x = optics.m_chrom->dmu(0)/2./UAL::pi;
  double chrm_y = optics.m_chrom->dmu(0)/2./UAL::pi;

  cout << "beta_x = " << beta_x << "  beta_y = " << beta_y << endl;
  cout << "Q_x = " << q_x << "  Q_y = " << q_y << endl;
  cout << "chrom_x = " << chrm_x << "  chrom_y = " << chrm_y << endl;
 
  double y0 = sqrt(amplit_y*beta_y/(6*gamma))*0.001 + tws.d(1)*dpp0;
  double y0p = tws.dp(1)*dpp0;
  double x0 = sqrt(amplit_x*beta_x/(6*gamma))*0.001 + tws.d(0)*dpp0;
  double x0p = tws.dp(0)*dpp0;

  for(int ip=0; ip < bunch.size(); ip ++){
    bunch[ip].getPosition().set(x0, x0p, y0, y0p, 0.0, dpp0);    
    // bunch[ip].setSpin(spin); // nikolay 1/15/10
  }

 // ************************************************************************
  std::cout << "\nTracking. " << std::endl;
  // ************************************************************************

  double t; // time variable
  
  //  int turns = 600000;   // 600000;
  int turns;
  configInput >> turns;

  std::cout << "\nTurns = " << turns << std::endl ;
  //return 0;

  std::string orbitFile = "./out/teapot/";
  orbitFile += variantName;
  orbitFile += ".orbit";

  PositionPrinter positionPrinter;
  positionPrinter.open(orbitFile.c_str());

  std::string spinFile = "./out/teapot/";
  spinFile += variantName;
  spinFile += ".spin";
  
  SpinPrinter spinPrinter;
  spinPrinter.open(spinFile.c_str());

  ba.setElapsedTime(0.0);

  start_ms();

  for(int iturn = 1; iturn <= turns; iturn++){

    ap -> propagate(bunch);
    
    for(int ip=0; ip < bunch.size(); ip++){
       positionPrinter.write(iturn, ip, bunch);
       // spinPrinter.write(iturn, ip, bunch); // nikolay 1/15/10
    }

  }

  t = (end_ms());
  std::cout << "time  = " << t << " ms" << endl;

  positionPrinter.close();
  // spinPrinter.close(); // nikolay 1/15/10

  return 1;
}

