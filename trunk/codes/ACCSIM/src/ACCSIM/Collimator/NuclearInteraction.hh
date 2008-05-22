// Library       : ACCSIM
// File          : ACCSIM/Collimator/NuclearInteration.hh
// Copyright     : see Copyright file
// Author        : F.W.Jones
// C++ version   : N.Malitsky 

#ifndef UAL_ACCSIM_COLLIMATOR_NUCLEAR_INTERACTION_HH
#define UAL_ACCSIM_COLLIMATOR_NUCLEAR_INTERACTION_HH

#include "PAC/Beam/Particle.hh"
#include "ACCSIM/Base/UniformGenerator.hh"
#include "ACCSIM/Collimator/IMaterialPropagator.hh"

namespace ACCSIM {

  /** Simulates nuclear interactions of particle in the collimator */

  class NuclearInteraction : public IMaterialPropagator {

  public:

    /** Constructor */
    NuclearInteraction();

    /** Destructor */
    virtual ~NuclearInteraction();

    /** Set beam parameters 
	@param m mass [GeV]
	@param energy total energy [GeV]
	@param charge charge
     */
    void setBeam(double m, double energy, double charge);

    /** Set material parameters 
	@param A atomic mass
	@param Z atomic number
	@param rho density of material [g/cm^3]
	@param radlength radiation length [m]
     */
    void setMaterial(double A, double Z, double rho, double radlength);

    /** Update particle coordinates 
	@param particle the PacParticle object
	@param iseed seed value
	@param l length of material
     */
    void update(PAC::Particle& particle, double l, int& iseed);

    /** Returns the reverse value of the mean free path */
    double getRlam() const; 

  private:

   
    double getElasticCS() const;
    double getInelasticCS() const;
    void makeElasticScattering(PAC::Position& pos, int &iseed);

  private:

    void calculateCSs(double A);
    void calculateN(double A, double rho);

  private:

    // Particle momentum
    double m_p;

    // Material parameters

    double m_A;
    double m_Z;
    double m_rho;

    double m_N;    // particles/cm^3
    double m_sige; // barn  = 1.e-24 cm^2
    double m_sigi; // barn  = 1.e-24 cm^2
    double m_eP;   // sige/(sige + sigi)

    ACCSIM::UniformGenerator m_uGenerator;
    
  };

}


#endif
