//# Library       : ACCSIM
//# File          : ACCSIM/Base/OrbitUniformGenerator.cc
//# Copyright     : see Copyright file
//# Original code : 
//# C++ version   : 

#include <math.h>
#include "ACCSIM/Base/OrbitUniformGenerator.hh"

// Constructor
ACCSIM::OrbitUniformGenerator::OrbitUniformGenerator()
{
}

// Returns a random deviate.
double ACCSIM::OrbitUniformGenerator::getNumber(int& idum)
{
  return ran1(idum);
}


double ACCSIM::OrbitUniformGenerator::ran1(int& idum)
{
  int j;
  long k; 
  static long iy=0; 
  static long iv[NTAB];
  float temp;
 
  if(idum <= 0 || !iy)
  {
    if (-(idum) < 1) idum=1;
    else idum = -(idum);
    for (j=NTAB+7; j>=0; j--) {
      k=(idum)/IQ;
      idum=IA*(idum-k*IQ) - IR*k;
      if (idum < 0) idum += IM;
      if (j< NTAB) iv[j] = idum;
    }
    iy=iv[0];
  }
 
  k=(idum)/IQ;
  idum=IA*(idum-k*IQ)-IR*k;
  if (idum < 0) idum += IM;
  j=iy/NDIV;
  iy=iv[j];
  iv[j]=idum;
  if ((temp=AM*iy) > RNMX) return RNMX; 
  else return temp;  
}
