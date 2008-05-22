/*  ******************************************************************************************
    *                                                                                        *
    *        CORAANT Library: COmprehensive Root-based Accelerator ANalysis Toolkit          * 
    *                                                                                        *
    *        Class TGNtupleD: It is a TNtupleD with the ablility to graph                    *
    *                                                                                        *
    *        Copyright   : see Copyright file                                                *
    *        Author      : Raymond Fliller III                                               *
    *                                                                                        *
    *                                                                                        *
    *                                                                                        *
    ******************************************************************************************
*/

#ifndef __TGNtupleD__
#define __TGNtupleD__

#include "TGraphAdd.hh"
#include "TNtupleD.h"
#include "TString.h"
#include "TCut.h"

namespace CORAANT{
  
  class TGNtupleD: public TNtupleD,public TGraphAdd{
   
  public:
    TGNtupleD():TNtupleD(){;}
    TGNtupleD(const char* name, const char* title, const char* varlist, Int_t bufsize = 32000):TNtupleD(name,title,varlist,bufsize){;}
    TGNtupleD(TGNtupleD &gntuple);
    virtual ~TGNtupleD();
    
    virtual TList    *GetListOfFriends() const  {return TNtupleD::GetListOfFriends();}
    virtual Int_t Draw(const char* varexp, const char* selection, Option_t* option, Int_t nentries = 1000000000, Int_t firstentry = 0) {return TNtupleD::Draw(varexp, selection, option, nentries, firstentry);}
    virtual Int_t     GetSelectedRows() {return TNtupleD::GetSelectedRows();}
    virtual Double_t *GetV1()   {return TNtupleD::GetV1();}
    virtual Double_t *GetV2()   {return TNtupleD::GetV2();}
    virtual Double_t *GetV3()   {return TNtupleD::GetV3();}
    virtual TObjArray *GetListOfLeaves() {return TNtupleD::GetListOfLeaves();}
    virtual Int_t GetEntry(Int_t entry = 0, Int_t getall = 0){return TNtupleD::GetEntry();}
    virtual Double_t  GetEntries() const   {return TNtupleD::GetEntries();}
    virtual TLeaf *FindLeaf(const char *name){return TNtupleD::FindLeaf(name);}
    
    
    
    ClassDef(TGNtupleD,1)

   };
};

#endif
