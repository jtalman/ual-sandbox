// Program     : PAC
// File        : PAC/Beam/Particle.hh
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#ifndef UAL_PAC_PARTICLE_HH
#define UAL_PAC_PARTICLE_HH

#include "UAL/Common/Probe.hh"
#include "PAC/Beam/Position.hh"
#include "PAC/Beam/Spin.hh"

namespace PAC {

  /**
     Represents the individual particle described by sets of attributes: position, spin, etc.
  */

  class Particle : UAL::Probe
    {
    public:

      /** Constructor */
      Particle();
  
      /** Copy constructor */
      Particle(const Particle& p);

      /** Destructor */
      virtual ~Particle();

      /** Copy operator */
      const Particle& operator = (const Particle& p);

      // Access method

      /** Sets id */
      void setId(int id) { m_id = id;}

      /** Returns id */
      int getId() const { return m_id; }

      /** Returns true if the particle is lost */
      bool isLost() const;
  
      /**  Returns the flag variable. */
      int getFlag() const;
  
      /** Sets the flag variable. */
      void setFlag(int v);

      /** Returns a reference to the Position object. */
      Position& getPosition();

      /** Returns a constant reference to the Position object. */
      const Position& getPosition() const;

      /** Sets the Position data. */ 
      void setPosition(const Position& position);

      /** Returns a pointer to the Spin object. */
      Spin* getSpin();

      /** Returns a constant reference to the Spin object. */
      const Spin& getSpin() const;

      /** Sets the Spin data. */ 
      void setSpin(const Spin& spin);

    protected:
      
      /** Id */
      int m_id;

      /** Flag */
      int  m_flag;

      /** Position */
      Position m_position;

      /** Spin */
      Spin* m_spin;

      /** Default spin */
      static Spin s_spin;

    private:

      void initialize(const Particle& p);
      void deleteSpin();

  };

}

#endif
