#include <fstream>

#include "timer.h"

#include "ACCSIM/Bunch/BunchGenerator.hh"

#include "SIMBAD/SC/LSCCalculatorFFT.hh"
#include "SIMBAD/SC/TSCCalculatorFFT.hh"
#include "SIMBAD/SC/TSCPropagatorFFT.hh"

#include "Shell.hh"

using namespace UAL;

int main(){

  bool status;

  UAL::Shell shell;

  // ************************************************************************
  std::cout << "\nDefine the space of Taylor maps." << std::endl;
  // ************************************************************************
  
  shell.setMapAttributes(Args() << Arg("order", 5)); 

  // ************************************************************************
  std::cout << "\nRead SXF file (lattice description)." << std::endl;
  // ************************************************************************
  
  status = shell.readSXF(Args() 
			 << Arg("file",  "../data/ring-Oct-2003.sxf") 
			 << Arg("print", "./ring-Oct-2003.sxf"));
  if(!status) exit(1);

  // ************************************************************************
  std::cout << "\nSelect lattice." << std::endl;
  // ************************************************************************

  status = shell.use(Args() << Arg("lattice", "rng"));
  if(!status) exit(1);

  // ************************************************************************
  std::cout << "\nDefine beam attributes." << std::endl;
  // ************************************************************************

  double energy = 3.0 + UAL::pmass;
  shell.setBeamAttributes(Args() << Arg("energy", energy));

  // ************************************************************************
  std::cout << "\nRead ADXF file (propagator description). " << std::endl;
  // ************************************************************************

  status = shell.readAPDF(Args() << Arg("file", "../data/simbad.apdf"));
  if(!status) exit(1);

  // ************************************************************************
  std::cout << "\nDo linear analysis. " << std::endl;
  // ************************************************************************

  PacTwissData twiss;
  status = shell.analysis(Args() 
			  << Arg("print", "./analysis") 
			  << Arg("twiss", twiss)); 
  if(!status) exit(1);

  double qx = twiss.mu(0)/(2.*UAL::pi);
  double qy = twiss.mu(1)/(2.*UAL::pi);

  int   iqx = qx;
  int   iqy = qy;

  std::cout << "  qx = "   << qx << ", qy = " << qy << std::endl;

  double betax = twiss.beta(0);
  double betay = twiss.beta(1);

  std::cout << "  betax = " << betax << ", betay = " << betay << std::endl;    
 

  // ************************************************************************
  std::cout << "\nPrepare a bunch of particles. " << std::endl;
  // ************************************************************************ 

  int    np   = 1000;
  double nppb = 3.3e14/8;

  double ex   = 54.0e-6; // m*rad
  double ey   = 54.0e-6; // m*rad

  double suml    = 1567.;
  double harmon  = 9;
  double gamma   = energy/UAL::pmass;
  double v0byc   = sqrt(energy*energy - UAL::pmass*UAL::pmass)/energy;
  double halfCT  = suml/harmon/v0byc/4.;
  double halfDE  = 0.007;

  int    seed = -100;

  // Bunch parameters

  PAC::BeamAttributes& ba = shell.getBeamAttributes();
  ba.setMacrosize(nppb/np);

  PAC::Bunch bunch(np);
  bunch.setBeamAttributes(ba);

  // Bunch distribution

  ACCSIM::BunchGenerator bunchGenerator;
  PAC::Position emittance, halfWidth;

  // Transverse distribution

  double mFactor = 3;

  emittance.set(ex, 0.0, ey, 0.0, 0.0, 0.0);
  bunchGenerator.addBinomialEllipses(bunch, mFactor, twiss, emittance, seed);

  // Longitudinal distribution
  // Default: ACCSIM idistl = 4 : uniform in phase,Gaussian in energy

  halfWidth.set(0.0, 0.0, 0.0, 0.0, halfCT, halfDE);
  bunchGenerator.addUniformRectangles(bunch, halfWidth, seed);

  // ************************************************************************
  std::cout << "\nDefine SIMBAD SC Calculator. " << std::endl;
  // ************************************************************************

  int nxBins = 32;
  int nyBins = 32;
  double eps = 0.001;

  SIMBAD::TSCCalculatorFFT& tscFFT =  SIMBAD::TSCCalculatorFFT::getInstance(); 

  tscFFT.setMinBunchSize(200); 
  tscFFT.setMaxBunchSize(np);
  tscFFT.setEps(eps);
  tscFFT.setGridSize(nxBins, nyBins);

  SIMBAD::LSCCalculatorFFT& lscFFT =  SIMBAD::LSCCalculatorFFT::getInstance(); 
  lscFFT.setMaxCT(halfCT);

  // ************************************************************************
  std::cout << "\nTrack it. " << std::endl;
  // ************************************************************************ 

  shell.run(Args()
	    << Arg("turns", 10)
 	    << Arg("bunch", bunch));

  // ************************************************************************
  std::cout << "\nPrint results. " << std::endl;
  // ************************************************************************ 

  int ip;
  for(ip = 0; ip < bunch.size(); ip += 100){
    PAC::Position& pos = bunch[ip].getPosition();
    std::cout << ip 
	      << " , x = " << pos.getX() << ", y = " << pos.getY() 
	      << " , ct = " << pos.getCT() << std::endl;
  }

  /*  
  std::ofstream bunchFile("./bunch_out_new");
  char line[120];
  for(ip =0; ip < bunch.size(); ip++){
    PAC::Position& p = bunch[ip].getPosition();   
    sprintf (line, "i=%5d x=%14.8e px=%14.8e y=%14.8e py=%14.8e ct=%14.8e dE/p=%14.8e",
	     ip , p.getX(), p.getPX(), p.getY(), p.getPY(), p.getCT(), p.getDE()); 
    bunchFile << line << std::endl;
  }
  bunchFile.close();
  */
  
}

