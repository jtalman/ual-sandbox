#ifndef _SPINPRINTER_H
#define _SPINPRINTER_H

#include <fstream>
#include <math.h>

#include "PAC/Beam/Bunch.hh"
#include "UAL/UI/Shell.hh"
#include "SPINK/Propagator/SpinPropagator.hh"

class SpinPrinter
{
public:

    /** Constructor */
    SpinPrinter();

    void setLength(double suml);

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
    double m_ct0;
    double omega_sum;
    double omega_num;

};



#endif  /* _OUTPUTPRINTER_H */

