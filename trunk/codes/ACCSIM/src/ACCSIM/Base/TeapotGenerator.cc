//# Library       : ACCSIM
//# File          : ACCSIM/Base/TeapotGenerator.cc
//# Copyright     : see Copyright file
//# Original code : TEAPOT 
//# C++ version   : N.Malitsky 

#include <iostream>
#include <math.h>
#include "ACCSIM/Base/TeapotGenerator.hh"

// Constructor
ACCSIM::TeapotGenerator::TeapotGenerator(int seed)
{
  // FTPOT parameters

  nd_ = 21;
  nj_ = 24;
  nr_ = 55; 

  // On Suns (at least) this gets evaluated in real precision rather
  // than double, resulting in the rather poor excuse of 9.9999997171807d-10
  // for 1e-9.  So we hardwire the right value.

  scale_ = 1.0e-9;

  irnSize_ = nr_;
  irn_ = new int[irnSize_];

  init55(seed);
}

// Destructor
ACCSIM::TeapotGenerator::~TeapotGenerator()
{
  if(irn_) delete [] irn_;
}

// Get a random number, cutoff at cut sigma
// FTPOT: getran(cut)
double ACCSIM::TeapotGenerator::getran(double cut)
{
  if(cut == 0) cut = 2;

  double gr1, gr2;
  while(1){
    ndrn(gr1, gr2);
    if(fabs(gr1) < cut) break;
  }
  return gr1;
}

// Set seed
void ACCSIM::TeapotGenerator::setSeed(int seed)
{
  init55(seed);
} 

// Get seed
int ACCSIM::TeapotGenerator::getSeed() const
{
  return seed_;
}

// FTPOT: init55(seed)
// Using the given seed value, initialize the pseudo-random number
// generator and load the array IRN with a set of random numbers in
//     [0, MAXINT).
// Al Russell, 85/03/13

void ACCSIM::TeapotGenerator::init55(int seed)
{
  seed_ = seed;

  int lmod1 = seed_/ACCSIM_TEAPOT_MAXINT;
  irn_[nr_-1] = seed_ - lmod1*ACCSIM_TEAPOT_MAXINT;

  int j = seed_;
  int k = 1;
  int lmod2, ii, i;
  for(i = 1; i < nr_; i++){
    lmod2 = (nd_*i)/nr_;
    ii = (nd_*i) - lmod2*nr_;
    irn_[ii - 1] = k;
    k = j - k;
    if(k < 0) k += ACCSIM_TEAPOT_MAXINT;
    j = irn_[ii - 1];   
  }

  // call irngen a few times to "warm it up".

  // NEXT - Set to 0, reflecting the loading of IRN with a new set
  // of pseudo-random integers.

  next_ = irngen();
  next_ = irngen();
  next_ = irngen(); 
}
 
// Get a random deviate
// FTPOT: irngen()
// Assuming the array IRN has been properly initialized by constructor
// with "IRN" pseudo-random numbers, generate the next "NR" elements
// in the pseudo-random sequence.  The function returns the value 0.
// The values generated are integers in the range [0, ACCSIM_TEAPOT_MAXINT).

// FTPOT routines IRNGEN and INIT55 are taken from D.E. Knuth, The Art of
// Computer Programming, Vol. 2 / Seminumerical Algorithms, second
// ed., 1981. See especially, pages 25-28 and 170-173.  Great care
// should be used if you are tempted to "improve" the routines by
// changing the parameters NR, NJ, ND, and their derivatives.
// Al Russell, 85/03/13

int ACCSIM::TeapotGenerator::irngen()
{
  int nrmj = nr_ - nj_;

  int i, j;
  for(i = 0; i < nj_; i++){
    j = irn_[i] - irn_[i + nrmj];
    if(j < 0) j += ACCSIM_TEAPOT_MAXINT;
    irn_[i] = j;
  }

  for(i = nj_; i < nr_; i++){
    j = irn_[i] - irn_[i - nj_];
    if(j < 0) j += ACCSIM_TEAPOT_MAXINT;
    irn_[i] = j;
  }

  return 0;

}

// FTPOT: fran()
// Select the next entry from a sequence of "NR" externally generated
// pseudo-random integers and convert it to a double precision number
// in [0,1).  If necessary, invoke the random number generator IRNGEN
// to generate a new set of "NR" pseudo-random integers
// Al Russell, 85/03/13

double ACCSIM::TeapotGenerator::fran()
{
   if (next_ == nr_) next_ = irngen();

   double result = scale_*irn_[next_];
   next_ +=  1;
   return result;
}


// FTPOT: ndrn(gr1, gr2)
//     Given a source of uniformly distributed pseudo-random numbers,
//     NDRN returns two double precision numbers from a gaussian
//     distribution with zero mean and unit sigma.

//     This subroutine is based on the polar method due to Box, et al.
//     and described in D.E. Knuth, The Art of Computer Programming, 2nd
//     edition, vol. 2, p 117-118.
void ACCSIM::TeapotGenerator::ndrn(double& r_gr1, double& r_gr2)
{
  double gr1, gr2, zzr;
  while(1){

    //   These subroutine calls assume the use of the portable random
    //   number generator routines.  Any other desired generator may
    //   equally well be used.
    //   Select two uniformly distributed numbers in [0,1):
    gr1 = fran();
    gr2 = fran();

    //   transform to be in [-1,1)x[-1,1):
    gr1 = 2*gr1 - 1;
    gr2 = 2*gr2 - 1;

    //   test if point falls in unit circle:
    zzr = gr1*gr1 + gr2*gr2;
    if (zzr <= 1.) break;

  }

  // transform accepted point to gaussian distribution:
  zzr = sqrt( - 2*log(zzr)/zzr);
  gr1 *= zzr;
  gr2 *= zzr;

  // get a new pair of numbers:
  r_gr1 = gr1;
  r_gr2 = gr2;
}
