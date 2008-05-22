// Library       : ACCSIM
// File          : ACCSIM/Collimator/IMaterialPropagator.hh
// Copyright     : see Copyright file
// Author        : F.W.Jones
// C++ version   : N.Malitsky 

#ifndef UAL_ACCSIM_IMATERIAL_PROPAGATOR_HH
#define UAL_ACCSIM_IMATERIAL_PROPAGATOR_HH

#include "PAC/Beam/Particle.hh"

namespace ACCSIM {

  /** Common interface of different collimator algorithm-classes */

  class IMaterialPropagator {

  public:

    /** Set beam parameters 
	@param m mass [GeV]
	@param energy total energy [GeV]
	@param charge charge
     */
    virtual void setBeam(double m, double energy, double charge) = 0;

    /** Set material parameters 
	@param A atomic mass
	@param Z atomic number
	@param rho density of material [g/cm^3]
	@param radlength radiation length [m]
     */
    virtual void setMaterial(double A, double Z, double rho, double radlength) = 0;

    /** Updates particle coordinates 
	@param particle the PacParticle object
	@param l length of material
	@param iseed seed value
     */
    virtual void update(PAC::Particle& particle, double l, int& iseed) = 0;

  };

}


#endif
