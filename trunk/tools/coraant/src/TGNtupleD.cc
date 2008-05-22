/*  ******************************************************************************************
    *                                                                                        *
    *        Class TGNtupleD: It is a TNtupleD with the ablility to graph                    *
    *                                                                                        *
    *                                                                                        *
    ******************************************************************************************
*/


#include "TDirectory.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TBuffer.h"
#include "TGNtupleD.hh"

using namespace std;

extern TROOT *gROOT;
extern TStyle *gStyle;
extern TDirectory *gDirectory;
extern TTree *gTree;
extern Int_t gDebug;

ClassImp(CORAANT::TGNtupleD);

CORAANT::TGNtupleD::TGNtupleD(TGNtupleD &gntuple)
{

}

CORAANT::TGNtupleD::~TGNtupleD()
{
//*-*-*-*-*-*-*-*-*-*-*NtupleD destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =================
 
}

void CORAANT::TGNtupleD::Streamer(TBuffer &b){TNtupleD::Streamer(b);}
