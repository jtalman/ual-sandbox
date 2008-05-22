/*  ******************************************************************************************
    *                                                                                        *
    *        CORAANT Library: COmprehensive Root-based Accelerator ANalysis Toolkit          * 
    *                                                                                        *   
    *        Class TGNtuple: It is a TNtuple with the ablility to graph                      *
    *                                                                                        *
    *        Copyright   : see Copyright file                                                *
    *        Author      : Raymond Fliller III                                               *
    *                                                                                        *
    *                                                                                        *
    *                                                                                        *
    ******************************************************************************************
*/

#ifndef __TGNtuple__
#define __TGNtuple__

#include "TGraphAdd.hh"
#include "TNtuple.h"
#include "TString.h"
#include "TCut.h"

namespace CORAANT{

  class TGNtuple: public TNtuple,public TGraphAdd{
    
  public:
    TGNtuple():TNtuple(){;}
    TGNtuple(const char* name, const char* title, const char* varlist, Int_t bufsize = 32000):TNtuple(name,title,varlist,bufsize){;}
    TGNtuple(TGNtuple &gntuple);
    virtual ~TGNtuple();
    
    virtual TList    *GetListOfFriends() const  {return TNtuple::GetListOfFriends();}
    virtual Int_t Draw(const char* varexp, const char* selection, Option_t* option, Int_t nentries = 1000000000, Int_t firstentry = 0) {return TNtuple::Draw(varexp, selection, option, nentries, firstentry);}
    virtual Int_t     GetSelectedRows() {return TNtuple::GetSelectedRows();}
    virtual Double_t *GetV1()   {return TNtuple::GetV1();}
    virtual Double_t *GetV2()   {return TNtuple::GetV2();}
    virtual Double_t *GetV3()   {return TNtuple::GetV3();}
    virtual TObjArray *GetListOfLeaves() {return TNtuple::GetListOfLeaves();}
    virtual Int_t GetEntry(Int_t entry = 0, Int_t getall = 0){return TNtuple::GetEntry();}
    virtual Double_t  GetEntries() const   {return TNtuple::GetEntries();}
    virtual TLeaf *FindLeaf(const char *name){return TNtuple::FindLeaf(name);}
    
    
    
    ClassDef(TGNtuple,1)

  };

};

#endif
