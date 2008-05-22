//# Library     : ACCSIM
//# File        : ACCSIM/Bunch/BunchAnalyzer.cc
//# Copyright   : see Copyright file
//# Author      : F.W.Jones
//# C++ version : N.Malitsky

#include "ACCSIM/Bunch/BunchAnalyzer.hh"

// Constructor

ACCSIM::BunchAnalyzer::BunchAnalyzer()
{
} 

// Destructor

ACCSIM::BunchAnalyzer::~BunchAnalyzer()
{
} 

void
ACCSIM::BunchAnalyzer::getRMS(/*in*/  const PAC::Bunch& bunch,
			     /*out*/ PAC::Position& orbit,
			     /*out*/ PacTwissData& twiss,
			     /*out*/ PAC::Position& rms)
{
  int i, j, size = 0;

  // Find a reference orbit

  orbit *= 0.0;

  for(i = 0; i < bunch.size(); i++) {
    if(bunch[i].getFlag() < 1) {
      size++;
      orbit += bunch[i].getPosition();
    }
  }

  orbit /= size;

  // Find standard deviations (rms)

  PAC::Position d, d2, dp;

  for(i = 0; i < bunch.size(); i++) {
    if(bunch[i].getFlag() < 1) {
      const PAC::Position& p = bunch[i].getPosition();
      d  = p;
      d -= orbit;
      for(j = 0; j < p.size(); j++){
        d2[j] += d[j]*d[j];
      }
      for(j = 0; j < p.size(); j += 2){
        dp[j] += d[j]*d[j+1];
      }
    }
  }

  d2 /= size;
  dp /= size;

  // Find transverse RMS

  for(j = 0; j < 4; j += 2){
    rms[j]= sqrt(d2[j]*d2[j+1] - dp[j]*dp[j]);
  } 

  // Find Twiss parameters

  twiss.beta(0)  =  d2.getX()/rms.getX();
  twiss.alpha(0) = -dp.getX()/rms.getX();

  twiss.beta(1)  =  d2.getY()/rms.getY();
  twiss.alpha(1) = -dp.getY()/rms.getY();

  // Find longitudinal RMS

  rms.setCT(sqrt(d2.getCT()));
  rms.setDE(sqrt(d2.getDE()));

}

