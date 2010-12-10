#ifndef UAL_DEF_HH
#define UAL_DEF_HH

/**
   UAL Element-Algorithm-Probe framework, authors: N.Malitsky, R.Talman.
 */

namespace UAL {

  /** Pi */
  const double pi = 3.1415926536;
  
  /** Proton mass [GeV] */
  const double pmass = 0.9382796;

  /** Electron mass [GeV] */
  const double emass = 0.5110340e-3;

  /** Proton gyromagnetic ratio  */
  const double pG = 1.7928456;

  /** Infinity */
  const double infinity = 1.0e+20;

  /** Speed of light [m/c] */
  const double clight = 2.99792458e+8;

  /** elementary charge - http://physics.nist.gov/cgi-bin/cuu/Value?e|search_for=elecmag_in*/
  /*  should verify with Handbook of Particle Physics*/
  const double elemCharge = 1.602176487e-19;

}

#endif
