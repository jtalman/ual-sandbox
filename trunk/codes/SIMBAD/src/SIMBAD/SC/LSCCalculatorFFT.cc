// Library       : SIMBAD
// File          : SIMBAD/SC/LSCCalculatorFFT.cc
// Copyright     : see Copyright file
// Author        : N.D'Imperio, A.Luccio et al.

#include <cmath>
#include <cfloat>
#include <fstream>

#include "UAL/Common/Def.hh"
#include "SIMBAD/SC/LFactors.hh"
#include "SIMBAD/SC/LSCCalculatorFFT.hh"

using namespace std;

SIMBAD::LSCCalculatorFFT* SIMBAD::LSCCalculatorFFT::s_theInstance = 0;

SIMBAD::LSCCalculatorFFT& SIMBAD::LSCCalculatorFFT::getInstance()
{
  if(s_theInstance == 0) {
    s_theInstance = new SIMBAD::LSCCalculatorFFT();
  }
  return *s_theInstance;
}

void SIMBAD::LSCCalculatorFFT::setMaxBunchSize(int size)
{
  m_maxBunchSize = size;

  m_bins.resize(m_maxBunchSize);
  m_fractBins.resize(m_maxBunchSize);
}

void SIMBAD::LSCCalculatorFFT::setGridSize(int lBins)
{
  m_nBins        = lBins;

  // Vector of longitudinal grid values 
  m_grid.resize(m_nBins + 1);

  // Vector of line space charge density
  m_rho.resize(m_nBins + 1);

  for(int j=0; j <= m_nBins; j++){
    m_grid[j]       = 0.0;
    m_rho[j]        = 0.0;
  } 

  return;
}


void SIMBAD::LSCCalculatorFFT::defineLFactors(const PAC::Bunch& bunch)
{

  // Define the longitudinal grid

  double dct = 2.*m_maxCT/m_nBins;

  int ict;
  for(ict = 0; ict <= m_nBins; ict++){
    m_grid[ict] = ict*dct;
  }

  for(ict = 0; ict <= m_nBins; ict++){
    m_rho[ict]  = 0.0;
  }

  // Calculate the longitudinal density profile

  // Bin the particles

  double ct;

  for(int ip = 0; ip < bunch.size(); ip++){

    if(bunch[ip].isLost()) continue;

    const PAC::Position& pos = bunch[ip].getPosition();

    // Phi Bin 

    ct = pos.getCT() + m_maxCT;
    ict = (int)(ct/dct) ;

    if(ict < 0) {
      std::cerr << "SIMBAD::LongSCIntegrator: particle's ct < -" << m_maxCT << std::endl;
      exit(1);
    }

    if(ict > m_nBins) {
      std::cerr << "SIMBAD::LongSCIntegrator: particle's ct > " << m_maxCT << std::endl;
      exit(1);
    }

    m_bins[ip] = ict;
    m_fractBins[ip] = (ct - m_grid[ict])/dct;

    // Bilinear binning

    m_rho[ict]      +=  (1. - m_fractBins[ip]);      
    m_rho[ict + 1]  +=  m_fractBins[ip];

  }

  // Wrap 1st and last bins:

  m_rho[0] += m_rho[m_nBins];
  m_rho[m_nBins] = m_rho[0];

  for(int it = 0; it < m_nBins; it++){
    std::cout << it << " " << m_rho[it] << std::endl;
  }

  // Calculate a longitudinal weighting factors used in transverse
  // space charge calculations:

  // Find beam size

  double ctMin =   DBL_MAX;;
  double ctMax =  -DBL_MAX;;

  int ip, bunchSize = 0;
  for(ip = 0; ip < bunch.size(); ip++){

    if(bunch[ip].isLost()) continue;

    const PAC::Position& pos = bunch[ip].getPosition();

    ct = pos.getCT();
    if(ct  < ctMin) ctMin = ct;
    if(ct  > ctMax) ctMax = ct;

    bunchSize++;
  } 

  SIMBAD::LFactors& lFactors =  SIMBAD::LFactors::getInstance(m_maxBunchSize); 

  double ctFactor = (ctMax - ctMin)/(bunchSize*dct);

  std::cout << "ctFactor = " << ctFactor << std::endl;

  double lFactor;
  for(ip = 0; ip < bunch.size(); ip++){

    lFactors.setElement(0.0, ip);

    if(bunch[ip].isLost()) continue;

    ict = m_bins[ip];
    lFactor  = m_rho[ict]*(1. - m_fractBins[ip]);
    lFactor += m_rho[ict+1]*m_fractBins[ip];
    lFactor *= ctFactor;
    lFactors.setElement(lFactor, ip);
  }  

}


SIMBAD::LSCCalculatorFFT::LSCCalculatorFFT()
{
  m_maxBunchSize = 0;

  setMaxCT(1.0);
  setGridSize(32);
}




