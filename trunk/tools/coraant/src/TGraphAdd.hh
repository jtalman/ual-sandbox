/*  ******************************************************************************************
    *                                                                                        *
    *        CORAANT Library: COmprehensive Root-based Accelerator ANalysis Toolkit          * 
    *                                                                                        *
    *        Class TGTree: This class adds functionality to TTree's.  It assumes that the    *
    *                      derived class will ultimitely inherit  from a TTree.              *
    *                                                                                        *
    *                                                                                        *
    *        Copyright   : see Copyright file                                                *
    *        Author      : Raymond Fliller III                                               *
    *                                                                                        *
    ******************************************************************************************
*/

#ifndef __TGraphAdd__
#define __TGraphAdd__

#include "TString.h"
#include "TList.h"
#include "TCut.h"
#include "TLeaf.h"
#include "TObjArray.h"

namespace CORAANT {

  class TGraphAdd{
  protected:
    
    TString LeafNames();
    
    TGraphAdd(){;}  //protected constructor to insure that this is NOT
                    //instatiated withoth some inheritance

  public:
   
    virtual ~TGraphAdd(){;};
    
    // These function add functionality to classes that derive from TTrees.
    virtual Int_t Graph(const char* varexp, TCut &selection, Option_t* option="AP", Int_t nentries = 1000000000, Int_t firstentry = 0);
    virtual Int_t Graph(const char* varexp, const char* selection="", Option_t* option="AP", Int_t nentries = 1000000000, Int_t firstentry = 0);
    virtual Int_t MultiGraph(const char *varexp, const char *selection="", Option_t *chopt="AP");
    virtual Int_t MultiGraph(const char *varexp, TCut &selection, Option_t *chopt="AP"); 
    
    
    // These functions have to be defined in derived classes
    virtual TList *GetListOfFriends() {return 0;}
    virtual Int_t Draw(const char* varexp, const char* selection, Option_t* option, Int_t nentries = 1000000000, Int_t firstentry = 0){return 0;}
    virtual Int_t GetSelectedRows() {return 0;}
    virtual Double_t* GetV1() {return 0;}
    virtual Double_t* GetV2() {return 0;}
    virtual Double_t* GetV3() {return 0;}
    virtual TObjArray* GetListOfLeaves() {return 0;}
    virtual Int_t GetEntry(Int_t entry = 0, Int_t getall = 0){return 0;}
    virtual Double_t  GetEntries(){return 0;}
    virtual TLeaf *FindLeaf(const char *name){return 0;}
   
    ClassDef(TGraphAdd,0)
    
   };
   

}; 
#endif

 
