#ifndef EMTEAPOT_MARKER_HH
#define EMTEAPOT_MARKER_HH

#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#include<iostream>
#include<iomanip>
#include"ETEAPOT/Integrator/BasicTracker.hh"

#include"EMTEAPOT/Integrator/genMethods/Vectors.h"
#include"EMTEAPOT/Integrator/genMethods/spinExtern"
#include"EMTEAPOT/Integrator/genMethods/designExtern"
#include"EMTEAPOT/Integrator/genMethods/bunchParticleExtern"

namespace EMTEAPOT {

  /** Marker tracker. */

 class marker : public ETEAPOT::BasicTracker {

  public:

#include"EMTEAPOT/Integrator/markerMethods/classMethods"
#include"EMTEAPOT/Integrator/markerMethods/propagate.method"

   fstream NikolayOut;
   static int turns;
   static int markerCount;
 };

}

#endif
