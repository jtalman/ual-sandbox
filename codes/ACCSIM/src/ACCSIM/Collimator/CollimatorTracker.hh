// Library       : ACCSIM
// File          : ACCSIM/Collimator/CollimatorTracker.hh
// Copyright     : see Copyright file
// Author        : F.W.Jones
// C++ version   : N.Malitsky 

#ifndef UAL_ACCSIM_COLLIMATOR_TRACKER_HH
#define UAL_ACCSIM_COLLIMATOR_TRACKER_HH

#include "PAC/Beam/Bunch.hh"
#include "PAC/Beam/Position.hh"

#include "ACCSIM/Base/BasicPropagator.hh"
#include "ACCSIM/Base/UniformGenerator.hh"
#include "ACCSIM/Base/UniformGenerator.hh"
#include "ACCSIM/Collimator/NuclearInteraction.hh"
#include "ACCSIM/Collimator/BetheBlochSlower.hh"
#include "ACCSIM/Collimator/FermiScatter.hh"

namespace ACCSIM {

  /** Propagates a bunch of particles through a collimator */

  class CollimatorTracker : public BasicPropagator {

  public:

    /** Constructor */
    CollimatorTracker();

    /** Copy constructor */
    CollimatorTracker(const CollimatorTracker& ct);

    /** Destructor */
    virtual ~CollimatorTracker();

    /** Returns false */
    bool isSequence() { return false; }

    /** Returns a deep copy of this object */
    UAL::PropagatorNode* clone();    

    /** Set lattice elements */
    void setLatticeElements(const UAL::AcceleratorNode& lattice, 
			    int i0, int i1, 
			    const UAL::AttributeSet& beamAttributes);


    /** Propagates a bunch of particles */
    void propagate(UAL::Probe& bunch); 

    /** Defines the collimator length */
    void setLength(double l);

    /** Returns the collimator length */
    double getLength() const;

    /** Defines the material 
	@param A atomic mass 
	@param Z atomic number
	@param rho density of the material [g/cm^3]
	@param radlength radiation length [m]
     */
    void setMaterial(double A, double Z, double rho, double radlength);

    /** Defines aperture parameters 
	@param shape aperture id of the APERTURE values
	@param a horizontal half-aperture 
	@param b vertical half-aperture
     */
    void setAperture(int shape, double a, double b);

    /** Set a seed for the random generator */
    void setSeed(int iseed);

    /** Returns the current seed value */
    int getSeed() const;

    /** Returns a number of lost particles */
    int getLostParticles() const;

  public:

    /** Aperture types */
    enum APERTURE {
      ELLIPSE = 0,
      RECTANGLE,
      XFLAT,
      YFLAT,
      SIZE
    };

  private:

    void update(PAC::Particle& particle, double l, int &iseed);

    bool checkAperture(const PAC::Position& pos) const;

    void trackThroughDrift(PAC::Position& pos, double& at) const;
    double getDriftStep(const PAC::Position& pos, double at) const;
    
    void trackThroughMaterial(PAC::Particle& part, double& at, double rlam);

  private:

    // Length

    double m_length;

    // Aperture parameters

    static double s_maxsize;

    int m_shape;
    double m_a;
    double m_b;

    // Material parameters

    double m_A;
    double m_Z;
    double m_rho;
    double m_radlength;

    // Algorithm parameters

    static double s_dstep;
    int m_iseed;

    double v0byc;
    double E0;
    double p0;
    double m0;

    int m_nLosts;

  private:

    void init();
    void copy(const CollimatorTracker& tracker);

  private:

    // Random number generator
    ACCSIM::UniformGenerator m_uGenerator; 
    // ACCSIM::OrbitUniformGenerator m_uGenerator;

    // Collection of collimator algorithms
    ACCSIM::NuclearInteraction m_nuclearInteraction;
    ACCSIM::BetheBlochSlower m_stopper;
    ACCSIM::FermiScatter m_scatter;

  };

  class CollimatorRegister{
  public:
    CollimatorRegister();
  };
    

}

#endif
