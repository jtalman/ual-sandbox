// Library     : Teapot
// File        : Math/TeapotRandomGenerator.h
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky


#ifndef TEAPOT_RANDOM_GENERATOR
#define TEAPOT_RANDOM_GENERATOR

#define TEAPOT_RANDOM_MAXINT 1000000000

class TeapotRandomGenerator
{
public:

  // Constructor
  TeapotRandomGenerator(int seed);

  // Destructor
  virtual ~TeapotRandomGenerator();

  // Get the current seed value
  int getSeed() const;

  // Set the seed value
  void setSeed(int seed);

  // Get a random number, cutoff at cut sigma
  double getran(int cut);

protected:

  //  Using the given seed value, initialize the pseudo-random number
  //  generator and load the array IRN with a set of random numbers in
  //  [0, MAXINT).
  void init55(int seed);

  // Assuming the array IRN has been properly initialized by INIT55
  // with "NR" pseudo-random numbers, generate the next "NR" elements
  // in the pseudo-random sequence.  The function returns the value 1.
  // The values generated are integers in the range [0, MAXINT). 
  int irngen();

  // Select the next entry from a sequence of "NR" externally generated
  // pseudo-random integers and convert it to a double precision number
  // in [0,1).  If necessary, invoke the random number generator IRNGEN
  // to generate a new set of "NR" pseudo-random integers
  double fran();    

  // Given a source of uniformly distributed pseudo-random numbers,
  // NDRN returns two double precision numbers from a gaussian
  // distribution with zero mean and unit sigma.
  void ndrn(double& r_gr1, double& r_gr2);


private:

  // the initial seed value
  int seed_;
 
  // array of random integrers
  int  irnSize_; 
  int* irn_;

  // NEXT - Pointer to the next entry in the list of pseudo-random
  // integers, IRN.  NEXT is reset to 0 whenever IRN is reloaded
  // by a reference to function IRNGEN.
  int next_;

  // FTPOT parameters
  int nd_;
  int nj_;
  int nr_;

  double scale_;
 
};

#endif
