

#ifndef _SPINPRINTER_H
#define	_SPINPRINTER_H

#include <fstream>
#include <math.h>

#include "PAC/Beam/Bunch.hh"
#include "UAL/UI/Shell.hh"

class SpinPrinter
{
public:

    /** Constructor */
    SpinPrinter();

    /** open file*/
    void open(const char* filename);

    void write(int iturn, int ip, PAC::Bunch& bunch);

    void calculateOmega();

    /** close file */
    void close();

 protected:

    double get_psp0(PAC::Position& p, double v0byc);

 protected:

    std::ofstream output;

    double m_vs0;

    int    m_turn0;
    double m_phase0;

};



#endif	/* _OUTPUTPRINTER_H */

