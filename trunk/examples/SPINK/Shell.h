
#ifndef _COSY_UAL1_SHELL_H
#define	_COSY_UAL1_SHELL_H

#include <list>
#include <map>

#include "SMF/PacGenElements.h"
#include "SMF/PacLattices.h"
#include "PAC/Beam/Position.hh"
#include "UAL/UI/Shell.hh"

namespace COSY { namespace UAL1 {

    class Shell : public UAL::Shell {
        public:

            Shell();

        public:

            // UAL::Shell interface 

            /** Selects lattice */
	    virtual bool use(const UAL::Arguments& arguments) ;

       public:

            // UAL::Shell extensions

	    void addN(const std::string& name, double value);
	    void addMadK1K2(const std::string& name, double k1, double k2);

            void calculateTwiss();

            void writeTwissToFile(const char* fileName);

       protected:

           void   calculateMaps(PacVTps& oneTurnMap,
                                 std::vector<PacVTps>& maps, int order,
                                 PAC::BeamAttributes& ba);
            
           void   calculateTwiss(const PacVTps& oneTurnMap,
			         const std::vector<PacVTps>& maps,
                                 std::vector<PacTwissData>& twiss);


       protected:

           /** selected lattice */
           PacLattice* p_lattice;

           /** calculated twiss functions */
           std::vector<PacTwissData> m_twissVector;

    };

  }
}

#endif	/* _COSY_UAL1_SHELL_H */
