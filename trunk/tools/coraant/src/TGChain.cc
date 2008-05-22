/*  ******************************************************************************************
    *                                                                                        *
    *        Class TGChain: It is a TChain with the ablility to graph                          *
    *                                                                                        *
    *                                                                                        *
    ******************************************************************************************
*/

#include "TDirectory.h"
#include "TBuffer.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TDirectory.h"
#include "TGChain.hh"

using namespace std;


extern TROOT *gROOT;
extern TStyle *gStyle;
extern TDirectory *gDirectory;
extern TTree *gTree;
extern Int_t gDebug;

ClassImp(CORAANT::TGChain);

CORAANT::TGChain::TGChain(TGChain &gtree)
{

}

CORAANT::TGChain::~TGChain()
{
//*-*-*-*-*-*-*-*-*-*-*Chain destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =================
 
}


void CORAANT::TGChain::Streamer(TBuffer &b){TChain::Streamer(b);}
