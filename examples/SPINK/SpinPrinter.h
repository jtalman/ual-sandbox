

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

    /** close file */
    void close();

 protected:

    std::ofstream output;

};



#endif	/* _OUTPUTPRINTER_H */

