#ifndef UAL_AIM_MONITOR_COLLECTOR_HH
#define UAL_AIM_MONITOR_COLLECTOR_HH

#include <fstream>
#include <list>
#include <map>

#include "PAC/Beam/Position.hh"
#include "AIM/BPM/Monitor.hh"

namespace AIM {

  /** Collector of Monitor turn-by-turn data */

  class MonitorCollector  {

  public:

    /** Returns the only instance of this class*/
    static MonitorCollector& getInstance();

    /** Registers BPM  */
    void registerBPM(Monitor* bpm);

    /** Returns a collection of bpm turn-by-turn data */
    std::map<int, Monitor*>& getAllData();

    /** Finds a fft spectrum of turn-by-turn data for the selected bpm */ 
    void fft(int bmpIndex, 
	     std::vector<double>& freq, 
	     std::vector<double>& hspec, 
	     std::vector<double>& vspec);

    /** Clears BPM containers */
    void clear();

    /** Writes BPM data into file */
    void write(const char* fileName);

    void fft(std::list<PAC::Position>& tbt, 
	     std::vector<double>& freq, 
	     std::vector<double>& hspec, 
	     std::vector<double>& vspec);

  private:

    // output file
    std::ofstream m_file;

    // turn-by-turn data of registered BPM's
    std::map<int, Monitor*> m_bpms;

    // Singleton
    static MonitorCollector* s_theInstance;   

  private:

    /** Constructor */
    MonitorCollector();

  };

}

#endif
