// Library       : AIM
// File          : AIM/BTF/BTFBpmCollector.hh
// Copyright     : see Copyright file
// Author        : N.Malitsky 


#ifndef UAL_AIM_BTF_BPM_COLLECTOR_HH
#define UAL_AIM_BTF_BPM_COLLECTOR_HH

#include <list>
#include "AIM/BTF/BTFSignal.hh"
#include "AIM/BTF/BTFSpectrum.hh"

namespace AIM {

  /** BPM data collector */

  class BTFBpmCollector  {

  public:

    /** Returns the only instance of this class*/
    static BTFBpmCollector& getInstance();

    /** Adds the bpm signal */
    void addSignal(BTFSignal& signal);

    /** Adds the horizontal spectrum */
    void addHSpectrum(BTFSpectrum& spec);

    /** Adds the vertical spectrum */
    void addVSpectrum(BTFSpectrum& spec);

    /** Returns the transverse signals */
    const std::list<AIM::BTFSignal>& getSignals();

    /** Returns the horizontal spectrum */
    const std::list<AIM::BTFSpectrum>& getHSpectrum(); 

    /** Returns the vertical spectrum */
    const std::list<AIM::BTFSpectrum>& getVSpectrum(); 

    /** Clears all containers */
    void clear();

  private:

    // Singleton
    static BTFBpmCollector* s_theInstance;  

    // transverse signal
    std::list<AIM::BTFSignal> m_signals;

    // h spectrum
    std::list<AIM::BTFSpectrum> m_hSpecs;

    // v spectrum
    std::list<AIM::BTFSpectrum> m_vSpecs;

  private:

    // Constructor 
    BTFBpmCollector();


  };

}

#endif
