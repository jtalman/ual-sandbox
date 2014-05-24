#ifndef UAL_DEF_HH
#define UAL_DEF_HH

/**
   UAL Element-Algorithm-Probe framework, authors: N.Malitsky, R.Talman.
 */

namespace UAL {
 /**  The source for these values is http://physics.nist.gov/cuu/Constants/index.html */

  /** Pi */
  const double pi = 3.1415926536;
  
  /** Proton mass [GeV] */
  const double pmass = 0.938272046;                      //  (23) GeV/c^2       0.9382796;
  /** PDG, particle data group, July 2010 Particle Physics Booklet, IOP Publishing*/

  /** Electron mass [GeV] */
  const double emass = 0.510998928e-3;

  /** Proton g factor  */
  const double pg = 5.585694713;

  /** Proton anomalous magnetic moment  */
  const double pG = 1.79284736;

  /** Proton frozen spin gamma (direct function of pg) */
  const double pFSG = 1.24810735;

  /** Infinity */
  const double infinity = 1.0e+20;

  /** Speed of light [m/c] */
  const double clight = 2.99792458e+8;

  /** elementary charge - http://physics.nist.gov/cgi-bin/cuu/Value?e|search_for=elecmag_in*/
  /* Proton/Electron charge [C] */
  const double elemCharge = 1.602176565e-19;             // (40) C
  /** PDG, particle data group, July 2010 Particle Physics Booklet, IOP Publishing*/

  /* Vacuum Permittivity [F/m] */
  const double vcmPerm = 8.854187817e-12;                // F/m

}

#endif
