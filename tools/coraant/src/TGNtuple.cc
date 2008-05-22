/*  ******************************************************************************************
    *                                                                                        *
    *        Class TGNtuple: It is a TNtuple with the ablility to graph                      *
    *                                                                                        *
    *                                                                                        *
    ******************************************************************************************
*/


#include "TDirectory.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TBuffer.h"
#include "TGNtuple.hh"

using namespace std;

extern TROOT *gROOT;
extern TStyle *gStyle;
extern TDirectory *gDirectory;
extern TTree *gTree;
extern Int_t gDebug;

ClassImp(CORAANT::TGNtuple);

CORAANT::TGNtuple::TGNtuple(TGNtuple &gntuple)
{

}

CORAANT::TGNtuple::~TGNtuple()
{
//*-*-*-*-*-*-*-*-*-*-*Ntuple destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =================
 
}

void CORAANT::TGNtuple::Streamer(TBuffer &b){TNtuple::Streamer(b);}
