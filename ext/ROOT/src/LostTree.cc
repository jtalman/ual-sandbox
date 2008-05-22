/* ****************************************************************************
   *                                                                          *
   *  This file contains a class which will be the "ROOT Container" for       *
   *  The Lost Collector.                                                     *
   *                                                                          *
   *                                                                          *
   *                                                                          *
   *                                                                          *
   *                                                                          *
   *  Author: Ray Fliller III                                                 *
   *  See Copyright file.                                                     *
   *                                                                          *
   *                                                                          *
   *                                                                          *
   *                                                                          *
   ****************************************************************************  */


#include <cstring>
#include <vector>
#include <algorithm>
#include <fstream>

#include "TH1.h"
#include "TDirectory.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TBuffer.h"
#include "TVirtualPad.h"

#include "SMF/PacLattice.h"
#include "UAL/SMF/AcceleratorNode.hh"
#include "TEAPOT/Integrator/LostCollector.hh"


#include "LostTree.hh"


using namespace std;

extern TROOT *gROOT;
extern TStyle *gStyle;
extern TDirectory *gDirectory;
extern TTree *gTree;
extern Int_t gDebug;

ClassImp(UAL::LostTree);

UAL::LostTree::LostTree(const char* name, const char* title, Int_t splitlevel):TTree(name,title,splitlevel)
{
  
  int i,j;
  TEAPOT::LostCollector& TheLost= TEAPOT::LostCollector::GetInstance();
  int Nparticles=TheLost.GetNLost();
  
  GrowBranches();
  
  if(!Nparticles){
    Warning("LostTree","There are no particles in the LostCollector");
    return;
  }
  
  for(i=0;i<Nparticles;i++){
    p_index= TheLost.GetParticleIndex(i);
    turn=TheLost.GetTurn(i);
    for(j=0;j<6;j++) position[j]=TheLost.GetPosition(i)[j];
    elem_index= TheLost.GetElemIndex(i); 
    strncpy(names,TheLost.GetElemName(i).c_str(),20);
    names[20]='\0';
    elem_location=TheLost.GetLocation(i);
    Fill();
  }
  
  SetEstimate(GetEntries());

}

UAL::LostTree::LostTree(const char* name, const char* title, const char *filename, Int_t splitlevel):TTree(name,title,splitlevel)
{

  ifstream input(filename);
  char buffer[1024];
  int i;

  GrowBranches();

  if(!input){
    Error("LostTree","Cannot open file %s",filename);
    return;
  }

  input.getline(buffer,1024);  //read the header line.
  
  while(!input.eof()){
    input>>p_index>>turn>>elem_index>>elem_location>>names;
    for(i=0;i<6;i++) input>>position[i];
    Fill();
  }

  input.close();

  SetEstimate(GetEntries());

}

void UAL::LostTree::GrowBranches()
{
  
  Branch("Particle_Index",&p_index,"p_index/I");
  Branch("Turn_Number",&turn,"turn/I");
  Branch("Element_Index",&elem_index,"e_index/I");
  Branch("Loss_Location",&elem_location,"s/F");
  Branch("Particle_Position",position,"x:x':y:y':ds:dp/F");
  Branch("Element_name",names,"name/C");
  
}

void UAL::LostTree::RegisterLattice(UAL::AcceleratorNode *lat)
{

  int i,nelements;
  int Nloc;
  vector<Float_t> locations;
  PacLattice *lattice=dynamic_cast<PacLattice *>(lat);

  if(lattice==NULL){
    Error("RegisterLattice","Did not register proper lattice");
    return;
  }

  if (s.GetSize()){
    Error("RegisterLattice","A lattice is already registered with %s",GetName());
    return;
  }

  //cycle though the lattice and get all unique locations
  Nloc=lattice->getNodeCount();
  locations.resize(Nloc);
  s.Set(Nloc+1);
  for(i=0;i<Nloc;i++) locations[i]=lattice->getNodeAt(i)->getPosition(); 
  sort(locations.begin(),locations.end());
  nelements=1;
  s[0]=locations[0];
  for(i=1;i<Nloc;i++){
    if(locations[i]!=s[nelements-1]){
      s[nelements]=locations[i];
      nelements++;
    }
  }
  s[nelements]=s[nelements-1]+1; //one extra space for overflow bin
  nelements++;
  s.Set(nelements); //reallocate the memory - does not loose array
}


Int_t UAL::LostTree::DrawLoss(const char* selection, Option_t* option, Int_t nentries, Int_t firstentry)
{

  int ret=Draw("s>>foo",selection,"goff",nentries,firstentry);
  Double_t *losses=GetV1();
  int num=GetSelectedRows();
  int i;
  TH1F *histo;
  TH1 *foo;
  
  cout<<"The return value is "<<ret<<endl;
  cout<<"THe number of particle is "<<num<<endl;

  if(ret<=0){
    Error("DrawLoss","Error drawing the losses");
    return ret;
  }

  foo=(TH1F *)(gDirectory->Get("foo"));
  foo->SetDirectory(0);

  histo=new TH1F("loss","Particle Loss Locations",s.GetSize()-1,s.GetArray());
  for(i=0;i<num;i++) histo->Fill(losses[i]);
  histo->GetXaxis()->SetTitle("s [m]");
  histo->GetXaxis()->CenterTitle();
  histo->GetYaxis()->SetTitle("Number of lost Particles");
  histo->GetYaxis()->CenterTitle();

  histo->Draw();

  gPad->SetLogy();

  delete foo;

  return ret;

}

void UAL::LostTree::Streamer(TBuffer &R__b)
{
   // Stream an object of class UAL::LostTree.
   if (R__b.IsReading()) {
      UAL::LostTree::Class()->ReadBuffer(R__b, this);
      SetBranchAddress("Particle_Index",&p_index);
      SetBranchAddress("Turn_Number",&turn);
      SetBranchAddress("Element_Index",&elem_index);
      SetBranchAddress("Loss_Location",&elem_location);
      SetBranchAddress("Particle_Position",position);
      SetBranchAddress("Element_name",names);
   } else {
      UAL::LostTree::Class()->WriteBuffer(R__b, this);
   }
}

