/*  ******************************************************************************************
    *                                                                                        *
    *        CORAANT Library: COmprehensive Root-based Accelerator ANalysis Toolkit          * 
    *                                                                                        *
    *        Class TGTree: It is a TTree with the ablility to graph                          *
    *                                                                                        *
    *        Copyright   : see Copyright file                                                *
    *        Author      : Raymond Fliller III                                               *
    *                                                                                        *
    *                                                                                        *
    ******************************************************************************************
*/

#ifndef __TGTree__
#define __TGTree__

#include "TGraphAdd.hh"
#include "TTree.h"
#include "TString.h"
#include "TCut.h"

namespace CORAANT{

  class TGTree: public TTree,public TGraphAdd{
    
  public:
    TGTree():TTree(){;}
    TGTree(const char* name, const char* title, Int_t splitlevel = 99):TTree(name,title,splitlevel){;}
    TGTree(TGTree &gtree);
    virtual ~TGTree();
    
    virtual TList    *GetListOfFriends() const  {return TTree::GetListOfFriends();}
    virtual Int_t Draw(const char* varexp, const char* selection, Option_t* option, Int_t nentries = 1000000000, Int_t firstentry = 0) {return TTree::Draw(varexp, selection, option, nentries, firstentry);}
    virtual Int_t     GetSelectedRows() {return TTree::GetSelectedRows();}
    virtual Double_t *GetV1()   {return TTree::GetV1();}
    virtual Double_t *GetV2()   {return TTree::GetV2();}
    virtual Double_t *GetV3()   {return TTree::GetV3();}
    virtual TObjArray *GetListOfLeaves() {return TTree::GetListOfLeaves();}
    virtual Int_t GetEntry(Int_t entry = 0, Int_t getall = 0){return TTree::GetEntry();}
    virtual Double_t  GetEntries() const   {return TTree::GetEntries();}
    virtual TLeaf *FindLeaf(const char *name){return TTree::FindLeaf(name);}
    
    ClassDef(TGTree,2)
    

  };

};



#endif

