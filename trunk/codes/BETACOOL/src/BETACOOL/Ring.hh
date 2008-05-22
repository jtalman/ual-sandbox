#ifndef UAL_BETACOOL_RING
#define UAL_BETACOOL_RING

#include "Main/Teapot.h"

/** Adapter to the BETACOOL code (collection of electron cooling, ibs, 
internal target algorithms, authors: Anatoly Sidorin, Alexander Smirnov, 
Grigori Trubnikov, 
<a href="http://lepta.jinr.ru/betacool.htm"> home page </a>
*/

namespace BETACOOL
{
  /** Adapter to containers with lattice functions*/

  class Ring 
  {

  public:

    /** Returns singleton */

    static Ring& getInstance(const char* fileName = 0);

    /** Calculates twiss parameters and sets Ring containers */
    void build(const char* latticeName);

    /** Returns the UAL lattice */
    PacLattice& getLattice() { return m_lattice; }

  private:

    // Constructor
    Ring(std::string& fileName);

  private:

    // Singleton
    static Ring* s_theInstance;

    // UAL selected lattice
    PacLattice m_lattice;
    Teapot m_teapot;    

  };

};


#endif
