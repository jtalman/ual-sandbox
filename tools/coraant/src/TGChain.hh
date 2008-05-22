/*  ******************************************************************************************
    *                                                                                        *
    *        CORAANT Library: COmprehensive Root-based Accelerator ANalysis Toolkit          *
    *                                                                                        *
    *        Class TGChain: It is a TChain with the ablility to graph                        *
    *                                                                                        *
    *        Copyright   : see Copyright file                                                *
    *        Author      : Raymond Fliller III                                               *
    *                                                                                        *
    *                                                                                        *
    ******************************************************************************************
*/

#ifndef __TGChain__
#define __TGChain__

#include "TGraphAdd.hh"
#include "TChain.h"
#include "TString.h"
#include "TCut.h"

namespace CORAANT{
  class TGChain: public TChain,public TGraphAdd{
 
  public:
    TGChain():TChain(){;}
    TGChain(const char* name, const char* title):TChain(name,title){;}
    TGChain(TGChain &gtree);
    virtual ~TGChain();
    
    virtual TList    *GetListOfFriends() const  {return TChain::GetListOfFriends();}
    virtual Int_t Draw(const char* varexp, const char* selection, Option_t* option, Int_t nentries = 1000000000, Int_t firstentry = 0) {return TChain::Draw(varexp, selection, option, nentries, firstentry);}
    virtual Int_t     GetSelectedRows() {return TChain::GetSelectedRows();}
    virtual Double_t *GetV1()   {return TChain::GetV1();}
    virtual Double_t *GetV2()   {return TChain::GetV2();}
    virtual Double_t *GetV3()   {return TChain::GetV3();}
    virtual TObjArray *GetListOfLeaves() {return TChain::GetListOfLeaves();}
    virtual Int_t GetEntry(Int_t entry = 0, Int_t getall = 0){return TChain::GetEntry();}
    virtual Double_t  GetEntries() const   {return TChain::GetEntries();}
    virtual TLeaf *FindLeaf(const char *name){return TChain::FindLeaf(name);}
    
    
    ClassDef(TGChain,2)
      
    };

};

#endif

