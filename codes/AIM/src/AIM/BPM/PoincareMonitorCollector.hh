#ifndef UAL_AIM_POINCARE_MONITOR_COLLECTOR_HH
#define UAL_AIM_POINCARE_MONITOR_COLLECTOR_HH

#include <fstream>
#include <list>
#include <map>

#include "PAC/Beam/Bunch.hh"
#include "AIM/BPM/PoincareMonitor.hh"

namespace AIM {

  /** Collector of PoincareMonitor turn-by-turn data */

  class PoincareMonitorCollector  {

  public:

    /** Returns the only instance of this class*/
    static PoincareMonitorCollector& getInstance();

    /** Registers BPM  */
    void registerBPM(PoincareMonitor* bpm);

    /** Returns a collection of bpm turn-by-turn data */
    std::map<int, PoincareMonitor*>& getAllData();

    /** Finds fft  of turn-by-turn data for the selected bpm */ 
    void fft(int bmpIndex, std::vector<double>& xFreq, std::vector<double>& yFreq);

    /** Clears BPM containers */
    void clear();

  protected:

    void fft(std::list<PAC::Bunch>& tbt,std::vector<double>& xFreq, std::vector<double>& yFreq);

  protected:

    // turn-by-turn data of registered BPM's
    std::map<int, PoincareMonitor*> m_bpms;

    // Singleton
    static PoincareMonitorCollector* s_theInstance;   

  private:

    /** Constructor */
    PoincareMonitorCollector();

  };

};

#endif
