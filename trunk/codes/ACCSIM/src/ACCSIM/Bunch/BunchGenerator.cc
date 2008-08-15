//# Library     : ACCSIM
//# File        : ACCSIM/Bunch/BunchGenerator.cc
//# Copyright   : see Copyright file
//# Author      : F.W.Jones
//# C++ version : N.Malitsky 

#include "ACCSIM/Bunch/BunchGenerator.hh"
#include <stdlib.h>

// Constructor

ACCSIM::BunchGenerator::BunchGenerator()
{
}

// Destructor

ACCSIM::BunchGenerator::~BunchGenerator()
{
}

// Shift bunch particles (e.g. because of injection bumps)
void 
ACCSIM::BunchGenerator::shift(/*inout*/ PAC::Bunch& bunch,
			     /*in*/ PAC::Position& kick)
{
  for(int i=0; i < bunch.size(); i++){
    bunch[i].getPosition() += kick;
  }
}

// Update the bunch distribution by uniformly populated rectangles.

void 
ACCSIM::BunchGenerator::addUniformRectangles(/*inout*/ PAC::Bunch& bunch, 
					    /*in*/ const PAC::Position& halfWidth,
					    /*inout*/ int& seed)
{
  int i, j;
  for(i = 0; i < halfWidth.size(); i++) {
    double delta = halfWidth[i];
    if(delta > 0.0) {
      for(j = 0; j < bunch.size(); j++){
	bunch[j].getPosition()[i] += uran(seed)*2.*delta -delta;
      }
    }
  }

  return;
}

// Update the bunch distribution by "gaussian" rectangles.

void 
ACCSIM::BunchGenerator::addGaussianRectangles(/*inout*/ PAC::Bunch& bunch, 
					    /*in*/ const PAC::Position& rms,
					     /*in*/ double cut,
					    /*inout*/ int& seed)
{
  ACCSIM::TeapotGenerator  generator(seed);

  int i, j;
  for(i = 0; i < rms.size(); i++) {
    double delta = rms[i];
    if(delta > 0.0) {
      for(j = 0; j < bunch.size(); j++){
	bunch[j].getPosition()[i] += generator.getran(cut)*delta;
      }
    }
  }

  return;
}

// Update the bunch distribution by uniformly populated ellipses. 

void 
ACCSIM::BunchGenerator::addUniformEllipses(/*inout*/ PAC::Bunch& bunch, 
					  /*in*/ const PAC::Position& halfWidth,
					  /*inout*/ int& seed)
{
  return;
}


// Update the bunch distribution by  uniformly populated ellipses.

void 
ACCSIM::BunchGenerator::addUniformEllipses(/*inout*/ PAC::Bunch& bunch,
					  /*in*/ const PacTwissData& twiss,
					  /*in*/ const PAC::Position& emittance,
					  /*inout*/ int& seed)
{
  return;
}

// Update the bunch distribution by  ellipses with binominal distribution.

void 
ACCSIM::BunchGenerator::addBinomialEllipses(/*inout*/ PAC::Bunch& bunch,
					   /*in*/ double m,
					   /*in*/ const PacTwissData& twiss,
					   /*in*/ const PAC::Position& emittance,
					   /*inout*/ int& seed)
{

  double gamma = 0.0, halfWidth1 = 0.0, halfWidth2 = 0.0;

  // Horizontal plane
  if(emittance.getX()) {

    if(!twiss.beta(0)) {
      cerr << "Error: ACCSIM::BunchGenerator : horizontal beta  == 0 \n";
      exit(1);
    }
    gamma = (1 + twiss.alpha(0)*twiss.alpha(0))/twiss.beta(0);

    halfWidth1 = sqrt(twiss.beta(0)*emittance.getX());
    halfWidth2 = sqrt(gamma*emittance.getX());

    addBinomialEllipse1D(bunch, 
			 m, 
			 0,  halfWidth1,
			 1, halfWidth2,
			 twiss.alpha(0), 
			 seed);
  }

  // Vertical plane
  if(emittance.getY()) {

    if(!twiss.beta(1)) {
      cerr << "Error: ACCSIM::BunchGenerator : vertical beta  == 0 \n";
      exit(1);
    }
 
    gamma = (1 + twiss.alpha(1)*twiss.alpha(1))/twiss.beta(1);

    halfWidth1 = sqrt(twiss.beta(1)*emittance.getY());
    halfWidth2 = sqrt(gamma*emittance.getY());

    addBinomialEllipse1D(bunch, 
			 m, 
			 2,  halfWidth1,
			 3, halfWidth2,
			 twiss.alpha(1), 
			 seed);
   
  }

  // Longitudinal plane

  if(emittance.getCT()*emittance.getDE()) {

    halfWidth1 = emittance.getCT()/2.;
    halfWidth2 = emittance.getDE()/2.;

    addBinomialEllipse1D(bunch, 
			 m, 
			 4, halfWidth1,
			 5, halfWidth2,
			 0.0, 
			 seed);
   
  }
}

// Add a binominal distribution in the index1/index2 plane. 

void 
ACCSIM::BunchGenerator::addBinomialEllipse1D(/*inout*/ PAC::Bunch& bunch,
					    /*in*/ double m,
					    /*in*/ int index1,
					    /*in*/ double halfWidth1,
					    /*in*/ int index2,
					    /*in*/ double halfWidth2,
					    /*in*/ double alpha,
					    /*inout*/ int& seed)
{
  if(index1 >= 6) return;
  if(index2 >= 6) return;

  // Number of particles
  int npart = bunch.size();

  double ran1, ran2, a, b, u, v, coschi, sinchi;

  for(int i = 0; i < npart; i++) {
    ran1 = uran(seed);
    ran2 = uran(seed);

    if(m == 0) a = 1.0;
    else       a = sqrt(1. - pow(ran1, 1./m));

    b = 2.*ACCSIM::pi*ran2;

    u = a*cos(b);
    v = a*sin(b);

    coschi = sqrt(1./(1.+alpha*alpha));
    sinchi = -alpha*coschi;

    bunch[i].getPosition()[index1] += u*halfWidth1;
    bunch[i].getPosition()[index2] += (u*sinchi + v*coschi)*halfWidth2;
  }
}


// Local random number generator

double
ACCSIM::BunchGenerator::uran(int& idum)
{
  return uniformGenerator_.getNumber(idum);
}




