#ifndef _XMGRACEPRINT_H
#define	_XMGRACEPRINT_H

#include <fstream>
#include <math.h>

#include "PAC/Beam/Bunch.hh"
#include "UAL/UI/Shell.hh"

class xmgracePrint
{
public:

    /** Constructor */
    xmgracePrint();

    /** open file*/
    void open(const char* filename);

    void write(int iturn, int ip, PAC::Bunch& bunch);

    /** close file */
    void close();

 protected:

     double get_psp0(PAC::Position& p, double v0byc);

 protected:

    std::ofstream output;

};



#endif	/* XMGRACEPRINT */
