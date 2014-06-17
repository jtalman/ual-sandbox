// Program     : PAC
// File        : PAC/Beam/BeamAttributes.hh
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#ifndef UAL_PAC_BEAM_ATTRIBUTES_HH
#define UAL_PAC_BEAM_ATTRIBUTES_HH

#include "UAL/Common/AttributeSet.hh"

namespace PAC {

  /** Collection of beam attributes (such as energy,charge, mass) common for Bunch's particles.
  */

  class BeamAttributes : public UAL::AttributeSet
    {
    public:

      // Constructors, destructor & copy operator

      /** Constructor */
      BeamAttributes();
  
      /** Copy constructor  */
      BeamAttributes(const BeamAttributes& ba);

      /** Destructor */
      virtual ~BeamAttributes();
 
      /** Copy operator */
      const BeamAttributes&  operator=(const BeamAttributes& ba);

      /** Returns a deep copy */
      UAL::AttributeSet* clone() const;

      // Access

      /** Returns the beam energy */
      double getEnergy() const;
  
      /** Sets the beam energy */
      void   setEnergy(double v);

      /**  Returns the mass of the particles in the beam.*/
      double getMass() const;
  
      /** Sets the mass of the particles in the beam. */
      void   setMass(double v);

      /** Returns the charge of the particles in the beam. */
      double getCharge() const;
  
      /** Sets the charge of the particles in the beam. */
      void   setCharge(double v);

      /** Sets the elapsed time */
      void setElapsedTime(double v);

      /** Returns the elapsed time */
      double getElapsedTime() const;

      /** Returns the revolution frequency. */
      double getRevfreq() const;
  
      /** Sets the revolution frequency. */
      void   setRevfreq(double v); 

      /** Returns the macrosize of the particles. */
      double getMacrosize() const;
  
      /** Sets the macrosize of the particles. */
      void   setMacrosize(double v); 

      /** Returns the gyromagnetic ratio of the particles. */
      double getG() const;
  
      /** Sets the gyromagnetic ratio of the particles. */
      void   setG(double v); 

      /** Returns the design angular momentum.*/
      double getL() const;
  
      /** Sets the design angular momentum. */
      void   setL(double v);

      /** Returns the design electric field.*/
      double getE() const;

      /** Sets the design electric field. */
      void   setE(double v);

      /** Returns the design "radius".*/
      double getR() const;

      /** Sets the design "radius". */
      void   setR(double v);

      /** Returns the g factor ratio of the particles. */
      double get_g() const;
  
      /** Sets the g factor ratio of the particles. */
      void   set_g(double v); 

    protected:

      /** energy */
      double m_energy;

      /** mass */
      double m_mass;

      /** charge */
      double m_charge;

      /** elapsed time */
      double m_time;

      /** revolution frequency */
      double m_revfreq;

      /** macrosize */
      double m_macrosize;

      /** gyromagnetic ratio */
      double m_G;      

      /** angular momentum */
      double m_L;

      /** electric field*/
      double m_E;

      /** "radius"*/
      double m_R;

      /** g factor */
      double m_gFac;      

    private:

      void initialize();
      void define(const BeamAttributes& ba);

    };

}

#endif
