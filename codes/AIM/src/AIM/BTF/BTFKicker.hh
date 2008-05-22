// Library       : AIM
// File          : AIM/BTF/BTFKicker.hh
// Copyright     : see Copyright file
// Author        : P.Cameron
// C++ version   : N.Malitsky 

#ifndef UAL_AIM_BTF_KICKER_HH
#define UAL_AIM_BTF_KICKER_HH

#include "AIM/BTF/BTFBasicDevice.hh"

namespace AIM {

  /** Kicker for Beam Transfer Function (BTF) measurement */

  class BTFKicker : public BTFBasicDevice  {

  public:

    /** Constructor */
    BTFKicker();

    /** Copy constructor */
    BTFKicker(const BTFKicker& kicker);

    /** Defines horizontal kick parameters */
    void setHKick(double hKick, double hNFreq, double hFracFreq, double hLag);

    /** Defines vertical kick parameters */
    void setVKick(double vKick, double vNFreq, double vFracFreq, double vLag);   

    /** Defines turn */
    void setTurn(int turn);

    /** Propagates a bunch of particles */
    void propagate(UAL::Probe& probe);

    /** Returns a deep copy of this node */
    UAL::PropagatorNode* clone();    

  protected:

    /** turn */
    int m_turn;

    /**  amplitude of horizontal kick */
    double m_hKick;

    /** integer number of horizontal frequency */
    double m_hNFreq;

    /** fraction of horizontal frequency */
    double m_hFracFreq;

    /** horizontal phase lag in multiples of 2pi */
    double m_hLag;

    /** amplitude of vertical kick */
    double m_vKick;

    /** integer number of vertical frequency */
    double m_vNFreq;

    /** fraction of vertical frequency */ 
    double m_vFracFreq;

    /** vertical phase lag in multiples of 2pi */
    double m_vLag;


  private:

    void init();
    void copy(const BTFKicker& kicker);

  };


}

#endif
