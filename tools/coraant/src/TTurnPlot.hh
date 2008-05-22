/*  ******************************************************************************************
    *                                                                                        *
    *        CORAANT Library: COmprehensive Root-based Accelerator ANalysis Toolkit          *
    *                                                                                        *
    *        Class TTurnPlot: This class is used to analyze the data from Fort.8 files       *
    *                          fort.8 format is defined on pg 39 of the fortran              *
    *                          TEAPOT manual                                                 *
    *                                                                                        *
    *        Copyright   : see Copyright file                                                *
    *        Author      : Raymond Fliller III                                               *
    *                                                                                        *
    *                                                                                        *
    ******************************************************************************************
*/

#ifndef __TTurnPlot__
#define __TTurnPlot__

#include "TGNtupleD.hh"
#include "TString.h"
#include "TCut.h"


namespace CORAANT {

  class TTurnPlot:public TGNtupleD{
    
  protected:
    
    TString date;
    TString time;
    Float_t version;
    Int_t seed;
    TString latticename;
    TString fort8name;
    Double_t betax,betay,alphax,alphay,Qx,Qy;
    Int_t nparts;
    Int_t nturns;

    int ReadFort8(const char *filename);
    
  public:
    TTurnPlot();
    TTurnPlot(const char *name,const  char *title,const char *filename);
    ~TTurnPlot(){;}
    
    
    //Getters 
    TString GetTime() const {return time;}
    TString GetDate() const {return date;}
    
    Float_t GetVersion() const {return version;}
    Int_t  GetSeed() const {return seed;}
    TString  GetLatticeName() const {return latticename;}
    Double_t GetBetaX() const {return betax;}
    Double_t GetBetaY() const {return betay;}
    Double_t GetAlphaX() const {return alphax;}
    Double_t GetAlphaY() const {return alphay;}
    Double_t GetQx() const {return Qx;}
    Double_t GetQy() const {return Qy;}
    TString GetFort8Name() const {return fort8name;}
    Int_t GetTurns() const {return nturns;}
    Int_t GetParticles() const {return nparts;}
    virtual Int_t TuneGraph(const char* plane, const TCut& selection, Option_t* option="AL"){return TuneGraph(plane,selection.GetTitle(),option);}
    virtual Int_t TuneGraph(const char* plane, const char* selection="", Option_t* option="AL");
    virtual Int_t TuneFootPrint(Option_t* option="SCAT");
    
    
    ClassDef(TTurnPlot,1)
  };

};
#endif
