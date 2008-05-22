/*  ******************************************************************************************
    *                                                                                        *
    *        Class TGTree: It is a TTree with the ablility to graph                          *
    *                                                                                        *
    *                                                                                        *
    ******************************************************************************************
*/

#include <cmath>
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <vector>
#include "TDirectory.h"
#include "TVirtualTreePlayer.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TH2.h"
#include "TH3.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TDirectory.h"
#include "TMultiGraph.h"
#include "TLegend.h"
#include "TFriendElement.h"
#include "TTree.h"
#include "TGraphAdd.hh"
#include "TPad.h"

using namespace std;

extern TROOT *gROOT;
extern TStyle *gStyle;
extern TDirectory *gDirectory;
extern TTree *gTree;
extern Int_t gDebug;

ClassImp(CORAANT::TGraphAdd);

TString CORAANT::TGraphAdd::LeafNames()
{
 
  TString names="";
  TIter Next(GetListOfFriends());
  TFriendElement *tree;
  TObjArray *leaves=GetListOfLeaves();
  int i;
  int size=leaves->GetSize();

  for(i=0;i<size;i++){
    if((TLeaf *)((*leaves)[i])){
      names+=((TLeaf *)((*leaves)[i]))->GetName();
      names+=" ";
    }
  }

  while(tree=(TFriendElement*)Next()){
    leaves=tree->GetTree()->GetListOfLeaves();
    size=leaves->GetSize();
    for(i=0;i<size;i++){
      if((TLeaf *)(*leaves)[i]){
	names+=tree->GetTreeName();
	names+=".";
	names+=((TLeaf *)(*leaves)[i])->GetName();
	names+=" ";
      }//close if
    } //close for
  }  //close while

  return names;
}



Int_t CORAANT::TGraphAdd::Graph(const char* varexp, TCut &selection, Option_t* option, Int_t nentries, Int_t firstentry)
{
  return Graph(varexp, selection.GetTitle(), option, nentries, firstentry);
}

Int_t CORAANT::TGraphAdd::Graph(const char* varexp, const char* selection, Option_t* option, Int_t nentries, Int_t firstentry)
{

  TGraph *graph;
  Int_t ret;
  char *vars, *ytitle, *title;
  TH2F *histo;
  TString errorx, errory;

  vars=new char [strlen(varexp)+20];
  
  sprintf(vars,"%s>>h",varexp);
  ret=Draw(vars,selection,"goff",nentries,firstentry);
 
  if(ret==-1){
    cout<<"cannot draw graph"<<endl;
    delete vars;
    return 0;
  }

  histo = (TH2F *)gDirectory->Get("h");
  histo->SetDirectory(0);
  
  ytitle=new char [strlen(varexp)+4]; 
 
  graph = new TGraph(GetSelectedRows(),GetV2(),GetV1());   
  graph->SetMarkerSize(0.9);
  graph->SetMarkerStyle(8);  
  graph->SetName(varexp);
  if(selection!=""){
    title=new char [strlen(varexp)+strlen(selection)+4];
    sprintf(title,"%s{%s}",varexp,selection);
    graph->SetTitle(title); 
    delete [] title;
  }else graph->SetTitle(varexp);
    
  if (strcmp(option,"goff")) graph->Draw(option);
  
  if(option!=""  || !strcmp(option,"goff")){
    strcpy(vars,"");
    strcpy(ytitle,"");
    strncat(ytitle,varexp,strchr(varexp,':')-varexp);
    strncat(vars,strchr(varexp,':')+1,strlen(varexp)-(strchr(varexp,':')-varexp));
    graph->GetYaxis()->SetTitle(ytitle);
    graph->GetXaxis()->SetTitle(vars);
    graph->GetXaxis()->CenterTitle();
    graph->GetYaxis()->CenterTitle();
  }


  delete [] ytitle;
  delete histo;
  if (vars) delete [] vars;
 
  return ret;
}


Int_t CORAANT::TGraphAdd::MultiGraph(const char *varexp, TCut &selection, Option_t *chopt)
{
  /* This function plots all varables in varexp in a TMultigraph.
     format of varexp is var1,var2,var3,var4,...varn:xvar
     where var1..varn are the varaibles to plot on the y axis
     and xvar is the variable on the xaxis.
     The other parameters are the same as Graph
  */
  return MultiGraph(varexp,selection.GetTitle(),chopt);
}

Int_t CORAANT::TGraphAdd::MultiGraph(const char *varexp, const char *selection, Option_t *chopt)
{
  /* This function plots all varables in varexp in a TMultigraph.
     format of varexp is var1,var2,var3,var4,...varn:xvar
     where var1..varn are the varaibles to plot on the y axis
     and xvar is the variable on the xaxis.
     The other parameters are the same as Graph
  */

  TMultiGraph *mg;
  TLegend *leg; 
  TGraph *graph=0;
  int i,nvars;
  int *pos;
  char *title, *ytitle;
  TString vars=varexp;
  TString graphvars;
  TString xvar;
  TString names=LeafNames();
  TString legname;

  nvars=0;
  pos = new int [vars.Length()];
  pos[0]=0;
  mg=0;

  for(i=0;i<vars.Length();i++){ //get indicies of commas
    if(vars[i]==','|| vars[i]==':' ){
      nvars++;
      pos[nvars]=i+1;
      if (vars[i]==':') break;    
    }
  }
   
  xvar=vars(pos[nvars],vars.Length()-pos[nvars]+1);
  if (!names.Contains(xvar)){
    cout<<xvar<<" is not a valid variable."<<endl;
    return 0;
  }
  if(selection!=""){
    title=new char [strlen(varexp)+strlen(selection)+4];
    sprintf(title,"%s{%s}",varexp,selection);
    mg=new TMultiGraph("mg",title);
    delete [] title;
  }else mg=new TMultiGraph("mg",varexp);
  leg= new TLegend(0.82,0.6,0.95,0.97,"Legend"); //only delete if graph not drawn.
  leg->SetFillColor(18);

  for(i=0;i<nvars;i++){  //loop over all varaibles
    graphvars=legname=vars(pos[i],pos[i+1]-pos[i]-1);
    graphvars+=":";
    graphvars+=xvar;     
    Graph(graphvars.Data(),selection);
    graph=(TGraph*)(gROOT->FindObject(graphvars.Data())); 
    if(graph){
      gPad->RecursiveRemove(graph);
      graph->SetMarkerColor(i+1);
      mg->Add(graph);
      leg->AddEntry(graph,legname.Data(),"P");
    }else cerr<<"MultiGraph:: Graph "<<graphvars<<" not drawn"<<endl;
    graph=0;
  }
  if(chopt!=""){
    ytitle=new char [strlen(varexp)+4]; 
    strcpy(ytitle,"");
    strncat(ytitle,varexp,strchr(varexp,':')-varexp);
    mg->Draw(chopt);
    mg->GetXaxis()->SetTitle(xvar.Data());
    mg->GetYaxis()->SetTitle(ytitle);
    mg->GetXaxis()->CenterTitle();
    mg->GetYaxis()->CenterTitle();
    leg->Draw();
    delete [] ytitle;
  }else delete leg;
  
  leg=0;
  delete [] pos;
  return 1;
}


