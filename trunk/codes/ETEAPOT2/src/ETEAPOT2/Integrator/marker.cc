#ifndef ETEAPOT2_MARKER_HH
#define ETEAPOT2_MARKER_HH

#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#include<iostream>
#include<iomanip>
#include"ETEAPOT/Integrator/BasicTracker.hh"

#include"ETEAPOT2/Integrator/genMethods/Vectors.h"
#include"ETEAPOT2/Integrator/genMethods/spinExtern"
#include"ETEAPOT2/Integrator/genMethods/designExtern"
#include"ETEAPOT2/Integrator/genMethods/bunchParticleExtern"

namespace ETEAPOT2 {

  /** Marker tracker. */

 class marker : public ETEAPOT::BasicTracker {

  public:

#include"ETEAPOT2/Integrator/markerMethods/classMethods"
#include"ETEAPOT2/Integrator/markerMethods/propagate.method"

   fstream NikolayOut;
   static int turns;
   static int markerCount;
 };

}

#endif
