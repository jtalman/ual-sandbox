/* ****************************************************************************
   *                                                                          *
   *  This file contains a class which will be the "ROOT Container" for       *
   *  The Lost Collector.                                                     *
   *                                                                          *
   *                                                                          *
   *                                                                          *
   *                                                                          *
   *                                                                          *
   *  Author: Ray Fliller III                                                 *
   *  See Copyright file.                                                     *
   *                                                                          *
   *                                                                          *
   *                                                                          *
   *                                                                          *
   ****************************************************************************  */

#ifndef __UAL_LostTree__
#define __UAL_LostTree__

#include "TTree.h"
#include "TArrayF.h"

namespace UAL{

  class AcceleratorNode;

  class LostTree:public TTree{
  private:
    
    Int_t p_index;  //contains particle index
    Int_t turn;     // turn number of loss
    Float_t position[6];  //particle position
    Int_t elem_index;           //element index
    Float_t elem_location;      //element s coordinate
    char names[21]; //names
    TArrayF s; //the s locations along lattice;

    void GrowBranches();


  public:

    LostTree():TTree(){;}
    LostTree(const char* name, const char* title, Int_t splitlevel = 99);  //initialize from global loss collector
    LostTree(const char* name, const char* title, const char *filename, Int_t splitlevel = 99);//initialize from file
    virtual ~LostTree(){;}

    void RegisterLattice(AcceleratorNode *lat);

    virtual Int_t DrawLoss(const char* selection="", Option_t* option="", Int_t nentries = 1000000000, Int_t firstentry = 0); 

    ClassDef(LostTree,1)

  };

};

  
#endif
