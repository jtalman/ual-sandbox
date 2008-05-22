
#include <fstream>
#include <vector>

#include "math.h"
#include "timer.h"

#include <rfftw.h>

#include "ACCSIM/Bunch/BunchGenerator.hh"

#include "SIMBAD/SC/LSCCalculatorFFT.hh"
#include "SIMBAD/SC/TSCCalculatorFFT.hh"
#include "SIMBAD/SC/TSCPropagatorFFT.hh"

#include "Shell.hh"

using namespace UAL;

int main(int argc, char** argv)
{

  bool status;

  UAL::Shell shell;

  std::ofstream tuneFile ("./tunes");

  // ************************************************************************
  std::cout << "\n1. Define the space of Taylor maps." << std::endl;
  // ************************************************************************
  
  shell.setMapAttributes(Args() << Arg("order", 5)); 

  // ************************************************************************
  std::cout << "\n2. Read SXF file (lattice description)." << std::endl;
  // ************************************************************************
  
  status = shell.readSXF(Args() 
			 << Arg("file",  "../data/ring-Oct-2003.sxf") 
			 << Arg("print", "./ring-Oct-2003.sxf"));
  if(!status) exit(1);

  // ************************************************************************
  std::cout << "\n3. Select lattice." << std::endl;
  // ************************************************************************

  status = shell.use(Args() << Arg("lattice", "rng"));
  if(!status) exit(1);

  // ************************************************************************
  std::cout << "\n4. Define beam attributes." << std::endl;
  // ************************************************************************

  double energy = 3.0 + UAL::pmass;
  shell.setBeamAttributes(Args() << Arg("energy", energy));

  // ************************************************************************
  std::cout << "\n5. Read ADXF file (propagator description). " << std::endl;
  // ************************************************************************

  status = shell.readAPDF(Args() << Arg("file", "../data/simbad.apdf"));
  if(!status) exit(1);

  // ************************************************************************
  std::cout << "\n6. Do linear analysis. " << std::endl;
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

  double alphax = twiss.alpha(0);
  double alphay = twiss.alpha(1);

  std::cout << "  alphax = " << alphax << ", alphay = " << alphay << std::endl;        

  // ************************************************************************
  std::cout << "\n7a. Prepare a bunch of particles. " << std::endl;
  // ************************************************************************ 

  int    np   = 10000;
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
  ba.setRevfreq(v0byc*UAL::clight/suml);

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
  // Default: ACCSIM idistl = 4 : uniform in phase

  halfWidth.set(0.0, 0.0, 0.0, 0.0, halfCT, halfDE);
  bunchGenerator.addUniformRectangles(bunch, halfWidth, seed);

  // ************************************************************************
  std::cout << "\n7d. Add prob particles. " << std::endl;
  // ************************************************************************ 

  int np_fft   =  np;
  int np_fm    =  90;
  int irs      =  10;
  int ias      =   9;

  std::vector<double> exi(np_fm);
  std::vector<double> eyi(np_fm);

  double gammax = (1. + alphax*alphax)/betax;
  double gammay = (1. + alphay*alphay)/betay;

  double xi, yi;
  int ip = 0;
  for(int ir = 0; ir < irs; ir++){
    for(int ia = 0; ia < ias; ia++){

      PAC::Position& pos = bunch[ip].getPosition();

      double rem  = (ir+1.0)/irs;
      double phi  = (ia+1.0)*(UAL::pi/2.)/(ias + 1);

      exi[ip] = rem*cos(phi);
      eyi[ip] = rem*sin(phi);
      xi      = sqrt(ex*exi[ip]/gammax);
      yi      = sqrt(ey*eyi[ip]/gammay);
      pos.set(xi, 0.0, yi, 0.0, 0.0, halfDE);
      ip++;
    }
  } 

  // ************************************************************************
  std::cout << "\n8. Define SIMBAD SC Calculator. " << std::endl;
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
  std::cout << "\n9. Estimate space charge tune-shift. " << std::endl;
  // ************************************************************************ 

  double r0 = 1.5410e-18;
  double Bf = (2.0*halfCT)/suml;

  double LaslettTuneShift = r0/4/UAL::pi;
  LaslettTuneShift       /= ex/4/1.5;
  LaslettTuneShift       *= nppb;
  LaslettTuneShift       /= v0byc*v0byc*gamma*gamma*gamma;
  LaslettTuneShift       /= Bf;

  std::cout << "   Laslett tune shift = " << LaslettTuneShift << std::endl;

  // ************************************************************************
  std::cout << "\n10. Track particles and accumulate its coordinates" 
	    << " for fft analysis" << std::endl;
  // ************************************************************************
 
  int nturns = 256; // 128; // 1024;

  fftw_real** xs    = new fftw_real*[np_fft];
  fftw_real** ys    = new fftw_real*[np_fft];
  fftw_real** xffts = new fftw_real*[np_fft];
  fftw_real** yffts = new fftw_real*[np_fft];

  fftw_real** xpower_spectrum = new fftw_real*[np_fft];
  fftw_real** ypower_spectrum = new fftw_real*[np_fft];

  for(ip = 0; ip < np_fft; ip++){
    xs[ip]    = new fftw_real[nturns];
    ys[ip]    = new fftw_real[nturns];
    xffts[ip] = new fftw_real[nturns];
    yffts[ip] = new fftw_real[nturns];

    xpower_spectrum[ip] = new fftw_real[nturns];
    ypower_spectrum[ip] = new fftw_real[nturns];
  }

  double t; // time variable
  start_ms();

  int it;
  for(it = 0; it < nturns; it++){

    shell.run(Args() << Arg("turns", 1) << Arg("bunch", bunch));

    for(ip =0; ip < np_fft; ip++){
      if(bunch[ip].isLost()) continue;
      PAC::Position& pos = bunch[ip].getPosition();
      xs[ip][it] = pos.getX();
      ys[ip][it] = pos.getY();
    }

    t = (end_ms());
    std::cout << "turn = " << it + 1 << ", time  = " << t << " ms" << endl;
  }

  // ************************************************************************
  std::cout << "\n11. FFT. " << std::endl;
  // ************************************************************************

  rfftw_plan fftplan = 
    rfftw_create_plan(nturns, FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE);

  for(ip = 0; ip < np_fft; ip++){

    if(bunch[ip].isLost()) continue;

    rfftw_one(fftplan, xs[ip], xffts[ip]);
    rfftw_one(fftplan, ys[ip], yffts[ip]);

    xpower_spectrum[ip][0] = xffts[ip][0]*xffts[ip][0];  /* DC component */
    ypower_spectrum[ip][0] = yffts[ip][0]*yffts[ip][0];  /* DC component */

    for (int k = 1; k < (nturns+1)/2; ++k){  /* (k < N/2 rounded up) */
      xpower_spectrum[ip][k] = 
	xffts[ip][k]*xffts[ip][k] + xffts[ip][nturns-k]*xffts[ip][nturns-k];
      ypower_spectrum[ip][k] = 
	yffts[ip][k]*yffts[ip][k] + yffts[ip][nturns-k]*yffts[ip][nturns-k];
    }
    if (nturns % 2 == 0) { /* N is even */
       xpower_spectrum[ip][nturns/2] = 
	 xffts[ip][nturns/2]*xffts[ip][nturns/2];  /* Nyquist freq. */
       ypower_spectrum[ip][nturns/2] = 
	 yffts[ip][nturns/2]*yffts[ip][nturns/2];  /* Nyquist freq. */
    }
  }

  double tunex = qx - iqx;
  double tuney = qy - iqy; 

  double xTune, xMinTune, xMaxTune; 
  double yTune, yMinTune, yMaxTune;

  xMinTune = yMinTune = +1.0;
  xMaxTune = yMaxTune = -1.0;

  char line [120];
  int lostParticles = 0;

  for(ip = 0; ip < np_fft; ip++){

    if(bunch[ip].isLost()) {
      std::cout << ip << " particle is lost: " 
		<< exi[ip] << " " 
		<< eyi[ip] << std::endl;
      lostParticles++;
      continue;
    }

    double ixfftmax   = 0;
    double iyfftmax   = 0;
    double xfftmax    = 0.0;
    double yfftmax    = 0.0;

    for(it = 0; it < (nturns+1)/2; it++){

      if(xfftmax < xpower_spectrum[ip][it]) {
	xfftmax  = xpower_spectrum[ip][it];
	ixfftmax = it;
      }
      if(yfftmax < ypower_spectrum[ip][it]) {
	yfftmax  = ypower_spectrum[ip][it];
	iyfftmax = it;
      }
    }

    double fft_tunex = ixfftmax/nturns;
    if(tunex > 0.5) fft_tunex = 1.0 - fft_tunex;

    double fft_tuney = iyfftmax/nturns;
    if(tuney > 0.5) fft_tuney = 1.0 - fft_tuney;

    //==========
    sprintf(line," %5d %14.8f %14.8f", ip, fft_tunex, fft_tuney) ;		 
    tuneFile << line << std::endl;		 		 
    //===========

    if(ip < np_fm) {

      xTune = fft_tunex - tunex;
      yTune = fft_tuney - tuney;

      std::cout << ip << " " << xTune << " " << yTune  << std::endl;

      if(xMaxTune < xTune) xMaxTune = xTune;
      if(xMinTune > xTune) xMinTune = xTune;
      if(yMaxTune < yTune) yMaxTune = yTune;
      if(yMinTune > yTune) yMinTune = yTune;
    }

  }

  std::cout << "Lost particles " << lostParticles << std::endl;

  std::cout << "xMaxTune = " << xMaxTune 
	    << ", xMinTune = " << xMinTune << std::endl;
  std::cout << "yMaxTune = " << yMaxTune 
	    << ", yMinTune = " << yMinTune << std::endl;

  rfftw_destroy_plan(fftplan);

  for(ip = 0; ip < np_fft; ip++){
    delete [] xs[ip];
    delete [] ys[ip];
    delete [] xffts[ip];
    delete [] yffts[ip];
    delete [] xpower_spectrum[ip];
    delete [] ypower_spectrum[ip];
  }

  delete [] xs;
  delete [] ys;
  delete [] xffts;
  delete [] yffts;
  delete [] xpower_spectrum;
  delete [] ypower_spectrum;

  tuneFile.close();


  
}

