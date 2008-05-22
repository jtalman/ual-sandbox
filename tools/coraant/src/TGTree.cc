/*  ******************************************************************************************
    *                                                                                        *
    *        Class TGTree: It is a TTree with the ablility to graph                          *
    *                                                                                        *
    *                                                                                        *
    ******************************************************************************************
*/

#include "TDirectory.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TDirectory.h"
#include "TGTree.hh"

using namespace std;

extern TROOT *gROOT;
extern TStyle *gStyle;
extern TDirectory *gDirectory;
extern TTree *gTree;
extern Int_t gDebug;

ClassImp(CORAANT::TGTree);

CORAANT::TGTree::TGTree(CORAANT::TGTree &gtree)
{

}

CORAANT::TGTree::~TGTree()
{
//*-*-*-*-*-*-*-*-*-*-*Tree destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =================
 
}

