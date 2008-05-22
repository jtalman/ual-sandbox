/* ****************************************************************************
   *                                                                          *
   *  This file contains functions which convert UAL containers into          *
   *  ROOT containers.                                                        *
   *                                                                          *
   *                                                                          *
   *                                                                          *
   *                                                                          *
   *                                                                          *
   *  Author: Ray Fliller III and Nikolay Malitsky                            *
   *  See Copyright file.                                                     *
   *                                                                          *
   *                                                                          *
   *                                                                          *
   *                                                                          *
   ****************************************************************************  */


#include "PAC/Beam/Bunch.hh"
#include "TNtupleD.h"

class TNtupleD;

namespace UAL {
  
  void bunch2Ntuple(const PAC::Bunch& bunch, TNtupleD& tnupled);
  

};
