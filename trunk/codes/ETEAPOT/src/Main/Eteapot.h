// Library     : Eteapot
// File        : Main/Eteapot.h
// Copyright   : see Copyright file
// Author      : John Talman

#ifndef ETEAPOT_H
#define ETEAPOT_H

#include <stdlib.h>

#include "SMF/PacSmf.h"
#include "PAC/Beam/Bunch.hh"

#include "Optics/PacChromData.h"
//#include "Integrator/EteapotElement.h"

#include "UAL/UI/Shell.hh"

class Eteapot : public PacSmf
{
public:
   void twissFromTracking( PAC::BeamAttributes ba, UAL::AcceleratorPropagator* ap, float m_m );
// void twissFromTracking( PAC::BeamAttributes ba, UAL::AcceleratorPropagator* ap, std::string filename );
// void twissFromTracking( PAC::BeamAttributes ba, UAL::AcceleratorPropagator* ap );
};
#endif
