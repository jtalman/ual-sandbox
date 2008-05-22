/*  ******************************************************************************************
    *                                                                                        *
    *        CORAANT Library: COmprehensive Root-based Accelerator ANalysis Toolkit          *
    *                                                                                        *
    *        Class TTuneGraph: This class is used to analyze tune data.  It contains         *
    *                          functions needed to denote peaks in a tune spectrum           *
    *                                                                                        *
    *                                                                                        *
    *        Copyright   : see Copyright file                                                *
    *        Author      : Raymond Fliller III                                               *
    *                                                                                        *
    *                                                                                        *
    ******************************************************************************************
*/

#ifndef __TTuneGraph__
#define __TTuneGraph__

#include "TAttLine.h"
#include "TGraph.h"

namespace CORAANT {

  class TTuneGraph:public TGraph{

  protected:
    
    TAttLine peaklines;

    Int_t npeaks;
    Float_t *positionsX; //[npeaks];
    Float_t *positionsY; //[npeaks];

    void SetDefaults();

  public:
    TTuneGraph(){SetDefaults();}
    TTuneGraph(Int_t n):TGraph(n){SetDefaults();}
    TTuneGraph(Int_t n, const Int_t* x, const Int_t* y):TGraph(n, x, y){SetDefaults();}
    TTuneGraph(Int_t n, const Float_t* x, const Float_t* y):TGraph(n, x, y){SetDefaults();}
    TTuneGraph(Int_t n, const Double_t* x, const Double_t* y):TGraph(n, x, y){SetDefaults();}
    TTuneGraph(const TVector& vx, const TVector& vy):TGraph(vx, vy){SetDefaults();}
    TTuneGraph(const TVectorD& vx, const TVectorD& vy):TGraph(vx, vy){SetDefaults();}
    TTuneGraph(const TH1* h):TGraph(h){SetDefaults();}
    TTuneGraph(const TF1* f, Option_t* option):TGraph(f,option){SetDefaults();}
    TTuneGraph(const char* filename, const char* format = "%lg %lg", Option_t* option=""):TGraph(filename,format,option){SetDefaults();}
    virtual ~TTuneGraph();

    virtual void PaintGraph(Int_t npoints, const Double_t* x, const Double_t* y, Option_t* option);

    virtual Int_t Search(Double_t sigma=1, Double_t threshold = 1e-4, Option_t *option="");
    const Float_t *GetPositionsX() const {return positionsX;}
    const Float_t *GetPositionsY() const {return positionsY;}
    const Int_t GetNPeaks() const  {return npeaks;}
    const TAttLine GetAttPeakLines() {return peaklines;}
    Float_t GetMaxPeakPositionX();
    Float_t GetMaxPeakPositionY();

    ClassDef(TTuneGraph,2)
     
  };

};

#endif
