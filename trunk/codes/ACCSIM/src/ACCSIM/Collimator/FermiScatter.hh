// Library       : ACCSIM
// File          : ACCSIM/Collimator/FermiScatter.hh
// Copyright     : see Copyright file
// Author        : F.W.Jones
// C++ version   : N.Malitsky

#ifndef UAL_ACCSIM_FERMI_SCATTER_HH
#define UAL_ACCSIM_FERMI_SCATTER_HH

#include "PAC/Beam/Particle.hh"
#include "ACCSIM/Base/GaussianGenerator.hh"
#include "ACCSIM/Collimator/IMaterialPropagator.hh"

namespace ACCSIM {

  /** Simulates the multiple Coulomb scattering based on the Fermi approach. */

  class FermiScatter : public IMaterialPropagator
  {
  public:

    /** Constructor */
    FermiScatter();

    /** Destructor */
    virtual ~FermiScatter();

    /** Set material parameters 
	@param A atomic mass
	@param Z atomic number
	@param rho density of material [g/cm^3]
	@param radlength radiation length [m]
     */
    void setMaterial(double A, double Z, double rho, double radlength);

    /** Returns the radiation length [m] */
    double getRadLength() const;

    /** Defines the beam parameters 
	@param m mass [GeV]
	@param energy total energy [GeV]
	@param charge charge
     */
    void setBeam(double m, double energy, double charge);

    /** Update particle coordinates
	@param particle the Particle object
	@param l length of the material [m]
	@param iseed seed value
     */  
    void update(PAC::Particle& particle, double l, int& iseed);

  private:

    // Returns the rms angular kick
    double getRmsAngle() const;

  private:

    // beam parameters

    double m_z; // charge of particle
    double m_rev_vp; // 1./beta/p

    // 1./sqrt(radlength)
    double m_radlength_factor;

  private:

    GaussianGenerator m_gaussGenerator;

  };

}

#endif
