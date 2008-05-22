// Library       : SIMBAD
// File          : SIMBAD/TSpaceCharge/TSCCalculatorFFT.cc
// Copyright     : see Copyright file
// Author        : N.D'Imperio, A.Luccio et al.

#include <cmath>
#include <cfloat>
#include <fstream>
#include <mpi.h>

#include "UAL/Common/Def.hh"
#include "SIMBAD/SC/LFactors.hh"
#include "SIMBAD/SC/LSCCalculatorFFT.hh"
#include "SIMBAD/SC_MPI/TSCCalculatorFFT_MPI.hh"

using namespace std;

SIMBAD::TSCCalculatorFFT_MPI* SIMBAD::TSCCalculatorFFT_MPI::s_theInstanceMPI = 0;

SIMBAD::TSCCalculatorFFT_MPI& SIMBAD::TSCCalculatorFFT_MPI::getInstance()
{
  if(s_theInstanceMPI == 0) {
    s_theInstanceMPI = new SIMBAD::TSCCalculatorFFT_MPI();
  }
  return *s_theInstanceMPI;
}

// Calculate the perveance
double SIMBAD::TSCCalculatorFFT_MPI::getPerveance(const PAC::Bunch& bunch) const
{
  // Find the real bunch size and its length
  
  double ctMin =  DBL_MAX;
  double ctMax = -DBL_MAX;

  double ct;
  int bunchCounter = 0;
  for (int ip = 0; ip < bunch.size(); ip++){

    if(bunch[ip].isLost()) continue;
      
    ct = bunch[ip].getPosition().getCT();
    if(ct  < ctMin) ctMin = ct;
    if(ct  > ctMax) ctMax = ct;
    bunchCounter++;
  }

  //Sync the ct and bunchCounter values;
  int bunchCounterGlo = 0;
  double ctMinGlobal = DBL_MAX;
  double ctMaxGlobal = -DBL_MAX;

  MPI_Allreduce(&bunchCounter, &bunchCounterGlo, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&ctMin, &ctMinGlobal, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&ctMax, &ctMaxGlobal, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  

  // Beam attributes

  double e = bunch.getBeamAttributes().getEnergy();
  double m = bunch.getBeamAttributes().getMass();
  double q = bunch.getBeamAttributes().getCharge();

  double beta  = sqrt(e*e - m*m)/e;
  double gamma = e/m;
  
  // Find the average line density "lambda" [nparticles/m]
  
  double bunchLength = fabs(ctMaxGlobal - ctMinGlobal)*beta;

  // cerr << "bunch length = " << bunchLength << "\n";

  if(bunchLength == 0){
    cerr << "TransSCIntegrator::getPerveance: bunchLength == 0 \n";
    exit(1);
  }

  double lambda = bunchCounterGlo*bunch.getBeamAttributes().getMacrosize()/bunchLength;

  double perveance  = q*q*lambda*rClassical*UAL::pmass;
  perveance        /= 2.*beta*beta*gamma*gamma*gamma*m;

  // cerr << "macrosize = " <<   bunch.getBeamAttributes().getMacrosize() << "\n";
  // cerr << "lambda = " << lambda << "\n";
  // cerr << "perveance = " << perveance << "\n";

  return perveance;

}

void SIMBAD::TSCCalculatorFFT_MPI::calculateForce(const PAC::Bunch& bunch)
{
  // if(bunch.size() < minBunchSize) return;
  
  // Initialize the grid and calculate the 2D density profile
  defineGridAndRho(bunch);

  // Calculate the FFT of the binned distribution: 
  fftBunchDensity();

  // Calculate the Green's funtion grid 
  calculateGreensGrid();

  // Calculate the FFT of the Greens Function.
  fftGreensGrid();

  // Calculate x and y forces 

  double perveance = getPerveance(bunch);
  int    bunchSize = getBunchSize(bunch);
  int globalBunchSize = 0;
  
  MPI_Allreduce(&bunchSize, &globalBunchSize, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  double factor = 4.0*perveance/(globalBunchSize*nXBins*nYBins);

  calculateForce(factor); 
}

void SIMBAD::TSCCalculatorFFT_MPI::defineGridAndRho(const PAC::Bunch& bunch)
{
  double xMax = -DBL_MAX;
  double xMin =  DBL_MAX;
  double yMax = -DBL_MAX;
  double yMin =  DBL_MAX;

  if(bunch.size() == 0) return;

  int i;
  double x, y;
  for(i = 0; i < bunch.size(); i++)
    {
      if(bunch[i].isLost()) return;

      const PAC::Position& p = bunch[i].getPosition();

      x = p.getX();

      if(x > xMax) xMax = x;
      if(x < xMin) xMin = x;

      y = p.getY();
      if(y > yMax) yMax = y;
      if(y < yMin) yMin = y;
      
    }

  //set global max and min

  double xMaxGlobal = -DBL_MAX;
  double xMinGlobal = DBL_MAX;
  double yMaxGlobal = -DBL_MAX;
  double yMinGlobal = DBL_MAX;

  MPI_Allreduce(&xMax, &xMaxGlobal, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&xMin, &xMinGlobal, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&yMax, &yMaxGlobal, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&yMin, &yMinGlobal, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

  

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

  double dxExtra = 0.5 * (xMaxGlobal-xMinGlobal);
  double dyExtra = 0.5 * (yMaxGlobal-yMinGlobal);

  double xGridMax = xMaxGlobal + dxExtra;
  double xGridMin = xMinGlobal - dxExtra;
  double yGridMax = yMaxGlobal + dyExtra;
  double yGridMin = yMinGlobal - dyExtra;

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
  for(int k = 0; k < (int)rho.size(); k++)
    {
      for(int n = 0; n < (int)rho[k].size(); n++)
	{
	  rho[k][n] = 0.0;
	}
    }

  // Zero bin containers

  for(i = 0; i < bunch.size(); i++)
    {
      xBin[i] = 0; 
      yBin[i] = 0;
      xFractBinPos[i] = 0.0;
      yFractBinPos[i] = 0.0;
    }

  double dxinv = 1.0/dx;
  double dyinv = 1.0/dy;

  int rhoSize = rho.size()*rho[0].size();
  std::vector<double> rhoLocal(rhoSize, 0);
  std::vector<double> rhoGlobal(rhoSize, 0);

  for(i=0; i < bunch.size(); i++)
    {

      if(bunch[i].isLost()) continue;

      const PAC::Position& p = bunch[i].getPosition();

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

  //copy rho to rhoLocal
  int rhoIndex = 0;
  for(i = 0; i < (int)rho.size(); i++)
    {
      for(int j = 0; j < (int)rho[i].size(); j++)
	rhoLocal[rhoIndex++] = rho[i][j];
    }

  MPI_Allreduce(&rhoLocal[0], &rhoGlobal[0], rhoSize,
		MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  
  //copy rhoGlobal to rho
  rhoIndex = 0;
  for(i = 0; i < (int)rho.size(); i++)
    {
      for(int j = 0; j < (int)rho[i].size(); j++)
	rho[i][j] = rhoGlobal[rhoIndex++];
    }

  
}
   

SIMBAD::TSCCalculatorFFT_MPI::TSCCalculatorFFT_MPI()
{
}





