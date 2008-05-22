// Library       : SIMBAD
// File          : SIMBAD/TSpaceCharge/TSCCalculatorFFT.cc
// Copyright     : see Copyright file
// Author        : N.D'Imperio, A.Luccio et al.

#include <cmath>
#include <cfloat>
#include <fstream>

#include "UAL/Common/Def.hh"
#include "SIMBAD/SC/LFactors.hh"
#include "SIMBAD/SC/LSCCalculatorFFT.hh"
#include "SIMBAD/SC/TSCCalculatorFFT.hh"

using namespace std;

double SIMBAD::TSCCalculatorFFT::rClassical = 1.534698e-18; // m
SIMBAD::TSCCalculatorFFT* SIMBAD::TSCCalculatorFFT::s_theInstance = 0;

SIMBAD::TSCCalculatorFFT& SIMBAD::TSCCalculatorFFT::getInstance()
{
  if(s_theInstance == 0) {
    s_theInstance = new SIMBAD::TSCCalculatorFFT();
  }
  return *s_theInstance;
}

// Find a bunch size
int SIMBAD::TSCCalculatorFFT::getBunchSize(const PAC::Bunch& bunch) const
{
   int result = 0;
   for(int j = 0; j < bunch.size(); j++){ 
     if(!bunch[j].isLost()) result++;
   }
   return result;
}

// Calculate the perveance
double SIMBAD::TSCCalculatorFFT::getPerveance(const PAC::Bunch& bunch) const
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

  // Beam attributes

  double e = bunch.getBeamAttributes().getEnergy();
  double m = bunch.getBeamAttributes().getMass();
  double q = bunch.getBeamAttributes().getCharge();

  double beta  = sqrt(e*e - m*m)/e;
  double gamma = e/m;
  
  // Find the average line density "lambda" [nparticles/m]
  
  double bunchLength = fabs(ctMax - ctMin)*beta;

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

void SIMBAD::TSCCalculatorFFT::setMinBunchSize(int size)
{
  minBunchSize = size;
}

int SIMBAD::TSCCalculatorFFT::getMinBunchSize() const
{
  return minBunchSize;
}


void SIMBAD::TSCCalculatorFFT::setMaxBunchSize(int size)
{
  maxBunchSize = size;

  xBin.resize(maxBunchSize);
  yBin.resize(maxBunchSize);

  xFractBinPos.resize(maxBunchSize);
  yFractBinPos.resize(maxBunchSize);

  SIMBAD::LSCCalculatorFFT::getInstance().setMaxBunchSize(size);
}

void SIMBAD::TSCCalculatorFFT::setGridSize(int nxb, int nyb)
{
  nXBins = nxb;
  nYBins = nyb;

  GreensF_re.resize(nXBins);
  GreensF_im.resize(nXBins);

  for(int i = 0; i < nXBins; i++)
    {
      GreensF_re[i].resize(nYBins);
      GreensF_im[i].resize(nYBins);
    }
 
  // Use this memory allocation scheme as per FFTW manual. Also
  // g++ chokes on new with arguments here (not everywhere!)
  in = (FFTW_COMPLEX *) 
    new char[nXBins * nYBins * sizeof(FFTW_COMPLEX)];
  fftRho =(FFTW_COMPLEX *) 
    new char[nXBins * nYBins * sizeof(FFTW_COMPLEX)];
  fftGF =(FFTW_COMPLEX *) 
    new char[nXBins * nYBins * sizeof(FFTW_COMPLEX)];
  fftForce =(FFTW_COMPLEX *) 
    new char[nXBins * nYBins * sizeof(FFTW_COMPLEX)];


  planForward = fftw2d_create_plan(nXBins, nYBins, 
				   FFTW_FORWARD, FFTW_MEASURE);
  planBackward = fftw2d_create_plan(nXBins, nYBins, 
				    FFTW_BACKWARD, FFTW_MEASURE);

  rho.resize(nXBins+2);
  for(int i = 0;i < nXBins+2; i++)
    {
      rho[i].resize(nYBins+2);
    }

  xGrid.resize(nXBins+1);
  yGrid.resize(nYBins+1);
		
		
  fscx.resize(nXBins+1);
  fscy.resize(nXBins+1);

  for(int i=0; i< nXBins+1; i++)
    {
      fscx[i].resize(nYBins+1);
      fscy[i].resize(nYBins+1);
    }

  return;
}

void SIMBAD::TSCCalculatorFFT::setEps(double epsilon)
{
  eps = epsilon;
}

void SIMBAD::TSCCalculatorFFT::calculateForce(const PAC::Bunch& bunch)
{
  if(bunch.size() < minBunchSize) return;
  
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
  double factor = 4.0*perveance/(bunchSize*nXBins*nYBins);

  calculateForce(factor); 
}


void SIMBAD::TSCCalculatorFFT::propagate(PAC::Bunch& bunch, double length)
{

  if(bunch.size() < minBunchSize) return;

  SIMBAD::LFactors& lFactors =  SIMBAD::LFactors::getInstance(maxBunchSize);  

  double f1, f2, f3, f4, fx, fy, lFactor;

  int j;
  for (j=0; j < bunch.size(); j++)
    {
      if(bunch[j].isLost()) continue;

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

      PAC::Position& pos = bunch[j].getPosition();

      double px = pos.getPX();
      double py = pos.getPY();

      px += fx * length * lFactor;
      py += fy * length * lFactor;

      pos.setPX(px);
      pos.setPY(py);  
    }

  return;
}

void SIMBAD::TSCCalculatorFFT::defineGridAndRho(const PAC::Bunch& bunch)
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

  // A. Fedotov and D. Abell: temporary change 
  /*
  double delta;
  delta = (xMax - xMin)/8;
  xMax += delta;
  xMin -= delta;

  delta = (yMax-yMin)/8;
  yMax += delta;
  yMin -= delta;
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
  
}
   
void SIMBAD::TSCCalculatorFFT::calculateGreensGrid()
{

  // Grid parameters

  double dx = 0;
  if(nXBins > 1) dx = xGrid[1] - xGrid[0];
  
  double dy = 0;
  if(nYBins > 1) dy = yGrid[1] - yGrid[0];


  // Calculate the Greens funtion grid FFT

  GreensF_re[nXBins/2][nYBins/2] = 0.0; // end point
  GreensF_im[nXBins/2][nYBins/2] = 0.0;

  double epsSq;
  double rTransX, rTransY, rTot2;

  if(eps < 0.0)
    epsSq = (eps)*(eps);
  else
    epsSq = (eps)*(eps) * dx * dy;

  int iX, iY;
  for (iY=1; iY <= (nYBins/2); iY++)   // assign middle to top-1 rows
    {
      rTransY = dy * (iY-1);
      for (iX = 1; iX <= nXBins/2; iX++)
	{
	  rTransX = (iX-1) * dx;
	  rTot2 = rTransX*rTransX + rTransY*rTransY + epsSq;
	  GreensF_re[iX-1][iY-1] = rTransX/rTot2;
	  GreensF_im[iX-1][iY-1] = rTransY/rTot2;
	}


      for (iX = nXBins/2+2; iX <= nXBins; iX++)
	{
	  rTransX = (iX - 1 - nXBins) * dx;
	  rTot2 = rTransX*rTransX + rTransY*rTransY + epsSq;
	  GreensF_re[iX-1][iY-1] = rTransX/rTot2;
	  GreensF_im[iX-1][iY-1] = rTransY/rTot2;
	}
    }


  for(iX=0; iX < nXBins; iX++)   // Null the top row:
    {
      GreensF_re[iX][nYBins/2] = 0.0;
      GreensF_im[iX][nYBins/2] = 0.0;
    }

  for(iY = nYBins/2+2; iY <= nYBins; iY++)  // Bottom rows:
    {
      rTransY = dy * (iY - 1 - nYBins);
      for (iX = 1; iX <= nXBins/2; iX++)
	{
	  rTransX = (iX-1) * dx;
	  rTot2 = rTransX*rTransX + rTransY*rTransY + epsSq;
	  GreensF_re[iX-1][iY-1] = rTransX/rTot2;
	  GreensF_im[iX-1][iY-1] = rTransY/rTot2;
	}

      for (iX = nXBins/2+2; iX <= nXBins; iX++)
	{
	  rTransX = (iX - 1 - nXBins) * dx;
	  rTot2 = rTransX*rTransX + rTransY*rTransY + epsSq;
	  GreensF_re[iX-1][iY-1] = rTransX/rTot2;
	  GreensF_im[iX-1][iY-1] = rTransY/rTot2;
	}
    }

}

void SIMBAD::TSCCalculatorFFT::fftGreensGrid()
{

  //   Calculate the FFT of the Greens Function:

  for(int j = 0; j < nYBins; j++)
    for(int i = 0; i < nXBins; i++)
      {
	c_re(in[nYBins*i+j]) = GreensF_re[i][j];
	c_im(in[nYBins*i+j]) = GreensF_im[i][j];
      }
    
  fftwnd(planForward, 1, in, 1, 0, fftGF, 1, 0);
}

void SIMBAD::TSCCalculatorFFT::fftBunchDensity()
{
  //   Calculate the FFT of the binned charge distribution:

  for(int j=0; j < nYBins; j++)
    for(int i=0; i  < nXBins; i++)
      {
	c_re(in[nYBins*i+j]) = rho[i][j];
	c_im(in[nYBins*i+j]) = 0.0;
      }
 
  fftwnd(planForward, 1, in, 1, 0, fftRho, 1, 0);
}

void SIMBAD::TSCCalculatorFFT::calculateForce(double factor)
{ 

  // Do Convolution:

  int index;
  for(int j = 0; j < nYBins; j++)
    for(int i = 0; i < nXBins; i++)

      {
	index = nYBins*i+j;
	c_re(in[index]) = c_re(fftRho[index])*c_re(fftGF[index]) -
	  c_im(fftRho[index])*c_im(fftGF[index]);
	c_im(in[index]) = c_re(fftRho[index])*c_im(fftGF[index]) +
	  c_im(fftRho[index])*c_re(fftGF[index]);
      }

  // Do Inverse FFT to get the Force:

  fftwnd(planBackward, 1, in, 1, 0, fftForce, 1, 0);

  int k, n;
  for(k = 0; k < (int)fscx.size(); k++)
    {
      for(n = 0; n < (int)fscx[k].size(); n++)
	{
	  fscx[k][n] = 0.;
	}
    }

  for(k = 0; k < (int)fscy.size(); k++)
    {
      for(n = 0; n < (int)fscy[k].size(); n++)
	{
	  fscy[k][n] = 0.;
	}
    }

  int iX, iY;
  for (iX = 1; iX <= nXBins; iX++)
    {
      for (iY = 1; iY <= nYBins; iY++)
	{
	  index = iY-1 + nYBins * (iX-1);
	  fscx[iX-1][iY-1] = c_re(fftForce[index]) * factor;
	  fscy[iX-1][iY-1] = c_im(fftForce[index]) * factor;

	}
    }
}

SIMBAD::TSCCalculatorFFT::TSCCalculatorFFT()
{
  init();
}

void SIMBAD::TSCCalculatorFFT::init()
{

  // Grid size

  nXBins = 0;
  nYBins = 0;

  // 

  minBunchSize = 1;
  maxBunchSize = 0;

  // Smoothing parmeter

  eps = 1.0;

  // FFT containers

  in       = 0;
  fftRho   = 0;
  fftGF    = 0;
  fftForce = 0;
}


SIMBAD::TSCCalculatorFFT::~TSCCalculatorFFT()
{
  if(in)
    {
      fftwnd_destroy_plan(planForward);
      fftwnd_destroy_plan(planBackward);

      delete [] in;        in = 0;
      delete [] fftRho;    fftRho = 0;
      delete [] fftGF;     fftGF = 0;
      delete [] fftForce;  fftForce = 0;
    }
}




// Print out the space charge force
void SIMBAD::TSCCalculatorFFT::showForce(char* f)
{
  std::ofstream file;
  file.open(f);
  if(!file) {
    cerr << "ORBIT_TransSCIntegrator::showForce: Cannot open " << f << " for output \n";
    exit(1);
  }
  char s[120];
  int iX, iY;
  file << "X and Y space charge forces : xBins = " << nXBins << " yBins = " << nYBins << "\n";
  file << "iX =  iY   =       xForce    =       yForce     =           rho \n";
  for (iY = 0; iY < nYBins; iY++){
    for (iX = 0; iX < nXBins; iX++){
      sprintf(s, "%-4d %-4d %-20.13e %-20.13e %-20.13e",
	   iX, iY,fscx[iX][iY],fscy[iX][iY],rho[iX][iY]);
      file << s << "\n";
    }
  }  

  file << "LFactors \n";
  SIMBAD::LFactors& lFactors =  SIMBAD::LFactors::getInstance(maxBunchSize); 

  for(int ip = 0; ip < lFactors.getSize(); ip++){
    file << ip << lFactors.getElement(ip) << std::endl;
  }

  file.close();
}
