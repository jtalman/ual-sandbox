// Library       : SIMBAD
// File          : SIMBAD/TSpaceCharge/TSCCalculatorFFT_3D_MPI.cc
// Copyright     : see Copyright file
// Author        : N.D'Imperio, A.Luccio et al.

#include <cmath>
#include <cfloat>
#include <fstream>
#include <mpi.h>

#include "UAL/Common/Def.hh"
#include "SIMBAD/SC/LFactors.hh"
#include "SIMBAD/SC/LSCCalculatorFFT.hh"
#include "SIMBAD/SC3D_MPI/TSCCalculatorFFT_3D_MPI.hh"
#include "SIMBAD/SC3D_MPI/LoadBalancer.hh"

using namespace std;

SIMBAD::TSCCalculatorFFT_3D_MPI* SIMBAD::TSCCalculatorFFT_3D_MPI::s_theInstanceMPI = 0;

SIMBAD::TSCCalculatorFFT_3D_MPI& SIMBAD::TSCCalculatorFFT_3D_MPI::getInstance()
{
  if(s_theInstanceMPI == 0) {
    s_theInstanceMPI = new SIMBAD::TSCCalculatorFFT_3D_MPI();
  }
  return *s_theInstanceMPI;
}

// Calculate the perveance
double SIMBAD::TSCCalculatorFFT_3D_MPI::getPerveance(PAC::Bunch& bunch,
						     vector<int>& subBunchIndices) const
{
  // Find the real bunch size and its length
  
  double ctMin =  DBL_MAX;
  double ctMax = -DBL_MAX;

  double ct;
  int bunchCounter = 0;
  int index;

  for (unsigned int ip = 0; ip < subBunchIndices.size(); ip++){
    index = subBunchIndices[ip];
    if(bunch[index].isLost()) continue;
      
    ct = bunch[index].getPosition().getCT();
    if(ct  < ctMin) ctMin = ct;
    if(ct  > ctMax) ctMax = ct;
    bunchCounter++;
  }

  //Beam attributes

  double e = bunch.getBeamAttributes().getEnergy();
  double m = bunch.getBeamAttributes().getMass();
  double q = bunch.getBeamAttributes().getCharge();

  double beta  = sqrt(e*e - m*m)/e;
  double gamma = e/m;
  
  // Find the average line density "lambda" [nparticles/m]

  //This needs to be checked, I don't think the beta factor is right NLD 7/16/04!!
  double bunchLength = SIMBAD::LoadBalancer::getInstance(bunch).getDeltaBeam()*beta;

  // cerr << "bunch length = " << bunchLength << "\n";

  if(bunchLength == 0){
    cerr << "TransSCIntegrator::getPerveance: bunchLength == 0 \n";
    exit(1);
  }

  double lambda = bunchCounter*bunch.getBeamAttributes().getMacrosize()/bunchLength;

  double perveance  = q*q*lambda*rClassical*UAL::pmass;
  perveance        /= 2.*beta*beta*gamma*gamma*gamma*m;

  // cerr << "macrosize = " <<   bunch.getBeamAttributes().getMacrosize() << "\n";
  // cerr << "lambda = " << lambda << "\n";
  // cerr << "perveance = " << perveance << "\n";

  return perveance;

}

void SIMBAD::TSCCalculatorFFT_3D_MPI::calculateForce(PAC::Bunch& bunch,
						     vector<int>& subBunchIndices)
{
  // if(bunch.size() < minBunchSize) return;

  // Initialize the grid and calculate the 2D density profile
  defineGridAndRho(bunch, subBunchIndices);

  // Calculate the FFT of the binned distribution: 
  fftBunchDensity();

  // Calculate the Green's funtion grid 
  calculateGreensGrid();

  // Calculate the FFT of the Greens Function.
  fftGreensGrid();

  // Calculate x and y forces 

  double perveance = getPerveance(bunch, subBunchIndices);
  int    bunchSize = subBunchIndices.size();
  double factor = 4.0*perveance/(bunchSize*nXBins*nYBins);

  SIMBAD::TSCCalculatorFFT::calculateForce(factor);

  return;
}

void SIMBAD::TSCCalculatorFFT_3D_MPI::propagate(PAC::Bunch& bunch,
						vector<int>& subBunchIndices,
						double length)
{

  if(bunch.size() < minBunchSize) return;

  SIMBAD::LFactors& lFactors =  SIMBAD::LFactors::getInstance(maxBunchSize);  

  double f1, f2, f3, f4, fx, fy, lFactor;

  unsigned int j;
  int index;
  for (j=0; j < subBunchIndices.size(); j++)
    {
      index = subBunchIndices[j];
      if(bunch[index].isLost()) continue;

      // Find horizontal force:

      f1 = fscx[xBin[j] - 1][yBin[j] - 1];
      f2 = fscx[xBin[j]][yBin[j] - 1];
      f3 = fscx[xBin[j]][yBin[j]];
      f4 = fscx[xBin[j] - 1][yBin[j]];

      fx = (1. - xFractBinPos[j]) * 
	(1. - yFractBinPos[j]) * f1 +
	xFractBinPos[j] * 
	(1. - yFractBinPos[j]) * f2 +
	(1. - xFractBinPos[j]) * 
	yFractBinPos[j] * f4 +       
	xFractBinPos[j] * 
	yFractBinPos[j] * f3;
      
      // Find Vertical Force:

      f1 = fscy[xBin[j] - 1][yBin[j] - 1];
      f2 = fscy[xBin[j]][yBin[j] - 1];
      f3 = fscy[xBin[j]][yBin[j]];
      f4 = fscy[xBin[j] - 1][yBin[j]];

          
      fy = (1. - xFractBinPos[j]) * 
	(1. - yFractBinPos[j]) * f1 +
	xFractBinPos[j] * 
	(1. - yFractBinPos[j]) * f2 +
	(1. - xFractBinPos[j]) * 
	yFractBinPos[j] * f4 +       
	xFractBinPos[j] * 
	yFractBinPos[j] * f3;

      // Include local line density effect, if it is available.

      lFactor = lFactors.getElement(j); // 1.0;

      PAC::Position& pos = bunch[index].getPosition();

      double px = pos.getPX();
      double py = pos.getPY();

      px += fx * length * lFactor;
      py += fy * length * lFactor;

      pos.setPX(px);
      pos.setPY(py);  
    }

  return;
}

void SIMBAD::TSCCalculatorFFT_3D_MPI::defineGridAndRho(const PAC::Bunch& bunch,
						       vector<int>& subBunchIndices)
{
  double xMax = -DBL_MAX;
  double xMin =  DBL_MAX;
  double yMax = -DBL_MAX;
  double yMin =  DBL_MAX;

  if(subBunchIndices.size() == 0) return;

  int index;
  double x, y;
  for(unsigned int i = 0; i < subBunchIndices.size(); i++)
    {
      index = subBunchIndices[i];
      if(bunch[index].isLost()) continue;

      const PAC::Position& p = bunch[index].getPosition();

      x = p.getX();

      if(x > xMax) xMax = x;
      if(x < xMin) xMin = x;

      y = p.getY();
      if(y > yMax) yMax = y;
      if(y < yMin) yMin = y;
      
    }

  // A. Fedotov and D. Abell: temporary change 
  /*
  double delta;
  delta = (xMaxGlobal - xMinGlobal)/8;
  xMaxGlobal += delta;
  xMinGlobal -= delta;

  delta = (yMaxGlobal-yMinGlobal)/8;
  yMaxGlobal += delta;
  yMinGlobal -= delta;
  */

  //define the grid extrema which is 2 times the actual particle extent

  double dxExtra = 0.5 * (xMax-xMin);
  double dyExtra = 0.5 * (yMax-yMin);

  double xGridMax = xMax + dxExtra;
  double xGridMin = xMin - dxExtra;
  double yGridMax = yMax + dyExtra;
  double yGridMin = yMin - dyExtra;

  //define the grid
  double dx = (xGridMax-xGridMin)/(double)nXBins;
  double dy = (yGridMax-yGridMin)/(double)nYBins;

  int iX;
  for (iX=0; iX<nXBins+1; iX++)
    xGrid[iX] = xGridMin + (double)(iX) * dx;

  int iY;
  for (iY=0; iY<nYBins+1; iY++)
    yGrid[iY] = yGridMin + (double)(iY) * dy;

  // Zero rho
  for(unsigned int k = 0; k < rho.size(); k++)
    {
      for(unsigned int n = 0; n < rho[k].size(); n++)
	{
	  rho[k][n] = 0.0;
	}
    }

  // Zero bin containers

  for(unsigned int i = 0; i < subBunchIndices.size(); i++)
    {
      xBin[i] = 0; 
      yBin[i] = 0;
      xFractBinPos[i] = 0.0;
      yFractBinPos[i] = 0.0;
    }

  double dxinv = 1.0/dx;
  double dyinv = 1.0/dy;

  for(unsigned int i=0; i < subBunchIndices.size(); i++)
    {
      index = subBunchIndices[i];
      if(bunch[index].isLost()) continue;

      const PAC::Position& p = bunch[index].getPosition();

      x = p.getX();
      y = p.getY();

      iX = 1 + int((x - xGridMin)*dxinv);
      iY = 1 + int((y - yGridMin)*dyinv);

      xBin[i] = iX; 
      yBin[i] = iY;

      xFractBinPos[i] = (x - xGrid[iX-1])*dxinv;
      yFractBinPos[i] = (y - yGrid[iY-1])*dyinv;

      // Bilinear binning:
      
      rho[iX-1][iY-1] += (1.0 - xFractBinPos[i]) * 
	(1.0 - yFractBinPos[i]);
      rho[iX-1][iY] += (1.0 - xFractBinPos[i]) * 
	yFractBinPos[i];
      rho[iX][iY-1] += xFractBinPos[i] * 
	(1.0 - yFractBinPos[i]);
      rho[iX][iY] += xFractBinPos[i] * yFractBinPos[i];
    }

  return;
}
   

SIMBAD::TSCCalculatorFFT_3D_MPI::TSCCalculatorFFT_3D_MPI()
{
}





