/*  ******************************************************************************************
    *                                                                                        *
    *        CORAANT Library: COmprehensive Root-based Accelerator ANalysis Toolkit          *
    *                                                                                        *
    *        Class TTuneSpace: This class is used to analyze tune data.  It plots the tune   *
    *                          space with resoance lines.                                    *
    *                                                                                        *
    *                                                                                        *
    *        Copyright   : see Copyright file                                                *
    *        Author      : Raymond Fliller III                                               *
    *                                                                                        *
    *                                                                                        *
    ******************************************************************************************
*/


#ifndef __TTuneSpace__
#define __TTuneSpace__

#include "TH2.h"


namespace CORAANT {

  class TTuneSpace:public TH2D{
  protected:

    Int_t order;         //the order of the resonances draw
    
  public:

    TTuneSpace();
    TTuneSpace(const char* name, const char* title, Int_t nbins=50);    
    TTuneSpace(const TTuneSpace& h2d);
    virtual ~TTuneSpace();

    virtual void Copy(TObject& hnew) const;
    virtual void Draw(Option_t* option="SCAT"){TH2D::Draw(option);}
    virtual void Paint(Option_t* option);
    
    const Int_t GetResonancesOrder() const {return order;}
    void SetResonancesOrder(Int_t o) {order=o;}
    
//     TTuneSpace&   operator=(const TTuneSpace &h1);
//     friend  TTuneSpace    operator*(Float_t c1, TTuneSpace &h1);
//     friend  TTuneSpace    operator*(TTuneSpace &h1, Float_t c1) {return operator*(c1,h1);}
//     friend  TTuneSpace    operator+(TTuneSpace &h1, TTuneSpace &h2);
//     friend  TTuneSpace    operator-(TTuneSpace &h1, TTuneSpace &h2);
//     friend  TTuneSpace    operator*(TTuneSpace &h1, TTuneSpace &h2);
//     friend  TTuneSpace    operator/(TTuneSpace &h1, TTuneSpace &h2);

    ClassDef(TTuneSpace,1)
  };

};

#endif
