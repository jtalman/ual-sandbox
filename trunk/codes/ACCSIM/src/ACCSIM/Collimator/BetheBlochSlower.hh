// Library       : ACCSIM
// File          : ACCSIM/Collimator/BetheBlochStower.hh
// Copyright     : see Copyright file
// Author        : F.W.Jones
// C++ version   : N.Malitsky 

#ifndef UAL_ACCSIM_BETHE_BLOCH_SLOWER_HH
#define UAL_ACCSIM_BETHE_BLOCH_SLOWER_HH

#include "PAC/Beam/Particle.hh"

#include "ACCSIM/Base/Def.hh"
#include "ACCSIM/Base/UniformGenerator.hh"
#include "ACCSIM/Collimator/IMaterialPropagator.hh"


namespace ACCSIM {

  /** Simulates the particle energy loss based on Bethe-Bloch approach. */

  class BetheBlochSlower : public IMaterialPropagator {

  public:

    /** Constructor */
    BetheBlochSlower();

    /** Destructor */
    virtual ~BetheBlochSlower();

    /** Sets material parameters 
	@param A atomic mass
	@param Z atomic number
	@param rho density of material [g/cm^3]
	@param radlength radiation length [m]
     */
    void setMaterial(double A, double Z, double rho, double radlength);

    /** Sets beam parameters 
	@param m  mass [GeV]
	@param energy total energy [GeV]
	@param charge charge
     */
    void setBeam(double m, double energy, double charge);

    /** Updates particle coordinates 
	@param particle the PacParticle instance
	@param l length of material [m]
	@param seed seed value
     */
    void update(PAC::Particle& particle, double l, int& seed);

  private: 

    // Returns the energy loss [GeV] */
    double getEnergyLoss(double meanLoss, double variance, double l, int& iseed);

    // Returns the mean energy loss divided by length */
    double getMeanEnergyLoss(PAC::Position& pos);

    // Returns the mean energy loss divided by length */
    double getMeanEnergyLoss();
    
    // Returns the variance divided by length */
    double getVarianceEnergyLoss();

    // Returns the energy loss parameter divided by length 
    double getEnergyLossParameter();    

  private:

    // Returns the maximum energy transferable to an atomic electron
    double getEmax();

    // Returns xi
    double getXi();

    // Returns the inverse cumulative Gaussian distribution 
    double getGaussIn(double r);

  private:

    UniformGenerator m_uGenerator;

  private:

    // Material parameters

    double m_A;
    double m_Z;
    double m_rho;

    // Beam parameters
    
    double m_m0;
    double m_energy;
    double m_beta;

  };

}


#endif
