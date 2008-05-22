// Program     : PAC
// File        : PAC/Beam/Bunch.hh
// Copyright   : see Copyright file
// Description:  Bunch is a vector of Particle's. 
// Author      : Nikolay Malitsky

#ifndef UAL_PAC_BUNCH_HH
#define UAL_PAC_BUNCH_HH

#include "UAL/Common/Probe.hh"
#include "PAC/Beam/BeamAttributes.hh"
#include "PAC/Beam/Particle.hh"

namespace PAC {

  /** Collection of Particle's. */

  class Bunch : public UAL::Probe
  {
  public:

    // Constructors & copy operator

    /** Constructor. The integer variable size defines 
	the number of particles in the bunch. */
    Bunch(int size = 0);
  
    /** Copy constructor. */
    Bunch(const Bunch& bunch);

    /** Destructor */
    virtual ~Bunch();

    /** Copy operator. */
    Bunch& operator=(const Bunch& bunch);

    /** Adds the bunch of particles */ 
    Bunch& add(const Bunch& bunch);

    /** Erases the ith particle */
    void erase(int i);

    // Access

    /** Set beam attributes */
    void setBeamAttributes(const BeamAttributes& ba);

    /** Returns beam attributes */
    PAC::BeamAttributes& getBeamAttributes();

    /** Returns beam attributes */
    const PAC::BeamAttributes& getBeamAttributes() const;

    /** Returns the number of particles. */
    int size() const;

    /** Resizes */
    void resize(int n);

    /** Returns the number od reserved particles */
    int capacity() const;

    /** Returns the particle specified by index. */
    Particle& getParticle(int index); 

    /** Sets the particle specified by index. */
    void setParticle(int index, const Particle& particle);     

    /** Returns the reference to the Particle object. */
    Particle& operator[](int index); 
  
    /** Returns the constant reference to the Particle object */
    const Particle& operator[](int index) const;

    /** Returns "order"-th bunch moments for all dimensions. */
    Position moment(int order) const;
  
    /** Returns the "order"-th bunch moment for the dimension specified by index. */
    double moment(int index, int order) const;

  protected :

    /** Beam attributes */
    PAC::BeamAttributes m_ba;

    /** Vector of reserved particles */
    std::vector<Particle> m_particles;

    /** Number of current particles */
    int m_size;

  };

}

#endif
        
