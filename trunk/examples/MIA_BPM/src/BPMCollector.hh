#ifndef UAL_MIA_BPM_COLLECTOR_HH
#define UAL_MIA_BPM_COLLECTOR_HH

#include <fstream>
#include <list>
#include <map>


#include "PAC/Beam/Position.hh"
#include "BPM.hh"

namespace MIA {

  /** Collector of BPM turn-by-turn data */

  class BPMCollector  {

  public:

    /** Returns the only instance of this class*/
    static BPMCollector& getInstance();

    /** Registers BPM  */
    void registerBPM(BPM* bpm);

    /** Returns a collection of bpm turn-by-turn data */
    std::map<int, BPM*>& getAllData();

    /** Clears BPM containers */
    void clear();

    /** Writes BPM data into file */
    void write(const char* fileName);

  private:

    // output file
    std::ofstream m_file;

    int m_hBPMs;
    int m_vBPMs;
    int m_hvBPMs; 

    // turn-by-turn data of registered BPM's
    std::map<int, BPM*> m_bpms;

    // Singleton
    static BPMCollector* s_theInstance;   

  private:

    /** Constructor */
    BPMCollector();

  };

};

#endif
