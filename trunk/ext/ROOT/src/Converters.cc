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


#include "Converters.h"

void UAL::bunch2Ntuple(const PAC::Bunch& bunch, TNtupleD& ntup)
{

  for(int i = 0; i < bunch.size();i++){
    //   if(bunch[i].isLost()) continue;
    const PAC::Position& pos = bunch[i].getPosition();
    ntup.Fill(pos.getX(),pos.getPX(),pos.getY(),pos.getPY(),pos.getCT(),pos.getDE(),bunch[i].getFlag());
  }

}
