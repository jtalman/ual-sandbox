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

#include <iostream>
#include <fstream>
#include <cstdio>
#include "TPad.h"
#include "TROOT.h"
#include "TDirectory.h"
#include "TH2.h"
#include "TGraph.h"
#include "TMultiGraph.h"
#include "TLegend.h"
#include "TAxis.h"
#include "TString.h"

#include "TTuneSpace.hh"
#include "TTuneGraph.hh"
#include "1DFFT.hh"

#include "TTurnPlot.hh"

using namespace std;

extern TROOT *gROOT;
extern TDirectory *gDirectory;

ClassImp(CORAANT::TTurnPlot);


CORAANT::TTurnPlot::TTurnPlot():TGNtupleD()
{

}

CORAANT::TTurnPlot::TTurnPlot(const char *name,const  char *title,const char *filename):TGNtupleD(name,title,"N:Turn:x:x':y:y':ct:dp")
{
  ReadFort8(filename);

  SetEstimate(GetEntries());
}


int CORAANT::TTurnPlot::ReadFort8(const char *filename)
{

  ifstream input(filename);
  const int buflength=1024;
  char buffer[buflength];
  int k;
  Double_t x[8];  //particle position (N:Turn:x:x':y:y':dp)

  if(!input.is_open()){
    Error("ReadFort8","File %s cannot be opened.", filename);
    return -1;
  }
  
  fort8name=filename;
  
  //parse first line
  input>>buffer;
  if(strcmp(buffer,"version")){
    Error("ReadFort8","File %s does not have fort.8 format", filename);
    return -1;
  }
  input>>version;
  input>>buffer;
  if(strcmp(buffer,"tracking")){
    Error("ReadFort8","File %s does not have fort.8 format", filename);
    return -1;
  }
  input>>buffer;
  date=buffer;
  input>>buffer;
  time=buffer;
  input>>seed;
  input.get(buffer[0]);

  //second line
  input.getline(buffer,buflength,'\n');
  latticename=buffer;

  //parse third line
  input>>nparts>>nturns;
  input>>betax>>betay>>alphax>>alphay>>Qx>>Qy;
  nturns++; //to be consistant with c++ conventions turn numbers run from 0->nturns
            //for a total of nturns++ turns.

  // parse the file
  if(version<4){
    x[0]=-1;
    input>>x[1];
    while(!input.eof()){
      if(x[1]==0){
	x[0]++;
	for(k=2;k<7;k++)input>>x[k]; 
	  x[8]=x[7]; //swap ct for dp
	  x[7]=0.0;
      }else for(k=2;k<6;k++)input>>x[k];
      Fill(x);
      input>>x[1];
    }
  }else{
    x[0]=-1;
    input>>x[1];
    while(!input.eof()){
      if(x[1]==0) x[0]++;
      for(k=2;k<8;k++)input>>x[k]; 
      Fill(x);
      input>>x[1];
    } //close while
  } //close else 
  input.close();

  return 1;

}


Int_t CORAANT::TTurnPlot::TuneGraph(const char* plane, const char* selection, Option_t* option)
{

  TTuneGraph *graph=0;
  TMultiGraph *mg=0;
  TLegend *leg=0;
  TH2F *histo;
  TAxis *xax=0, *yax=0;
  char choice[20],name[20];
  int i,j; 
  int Nparticles=0;  //number of particles to be plotted
  int *start; //index of the start of the ith particle in N
  char title[1024];
  Double_t *N,*f,*y, *spec; //arrays of particle number, tunes, and positions
  int n; //length of arrays above
  int size;
  real1DFFT *fft=NULL;
  double totmin=10e34;
  int oldpowersize=0;
  int oldsize=0;
  int npeaks=0;
  TString opt=option;

  //  cout<<"The selected plane is |"<<plane<<"|"<<endl;

  f=NULL;
  spec=NULL;

  if(strcmp(plane,"x") && strcmp(plane,"y")){
    Error("TuneGraph","Plane %s is not a valid plane.   Choose x or y",plane);
    return -1;
  }

  opt.ToLower();

  //"draw the histogram to get the data
  sprintf(choice,"N:%s>>h",plane);
  cout<<"About to draw the histogram"<<endl;
  Draw(choice,selection,"goff");
  //  cout<<"THe ClassName of drawn object is "<<gDirectory->Get("h")->ClassName()<<endl;
  histo = (TH2F *)gDirectory->Get("h");
  if(histo==NULL){
    Error("TuneGraph","Cannot get the histogram data.");
    return -1;
  }
  //  cout<<"The thing of located at "<<gDirectory->Get("h")<<endl;
  //  cout<<"And h is "<<histo<<endl;
  histo->SetDirectory(0);
  N=GetV1();
  y=GetV2();
  n=GetSelectedRows();
  
  if(n==0){
    Error("TuneGraph","There is not data to plot");
    delete histo;
    return -1;
  }
    cout<<"There are "<<n<<" rows of the data"<<endl;

  //count the number of particles
  start=new int[GetParticles()];
  if(n>0) { Nparticles=1; start[0]=0;}
  //  cout<<"The 0th particle is at index 0"<<endl;
  //  cout<<"N[0] = "<<N[0]<<endl;
  j=1;
  for(i=1;i<n;i++){
    //    cout<<"N["<<i<<"] = "<<N[i]<<endl;
    if(N[i-1]!=N[i]){
      Nparticles++;
      start[j]=i;
      //            cout<<"The "<<j<<"th particle is at index "<<i<<endl;
      j++;
    }
  }
  start[j]=n;// the end of the array
  // cout<<"The end of the particle array is at index "<<n<<endl;
  // cout<<"There are "<<Nparticles<<" in the selection"<<endl;
  
  if(strcmp(selection,""))  sprintf(title,"Tune Graph, plane %s. {%s}",plane,selection);
  else sprintf(title,"Tune Graph, plane %s.",plane);
  if (Nparticles>1){
    mg=new TMultiGraph("mg",title);
    leg= new TLegend(0.82,0.6,0.95,0.97,"Legend"); //only delete if graph not drawn.
    leg->SetFillColor(18);
  }

  for(i=0;i<Nparticles;i++){ //loop over particles and do FFT
    cout<<"Computing FFT #"<<i<<endl;
    sprintf(name,"Particle:%i",(int) N[start[i]]);
    size=start[i+1]-start[i];
    // cout<<"THe size of the array  to be tranformed is "<<size<<endl;
    if(fft==NULL) fft=new real1DFFT(size,y+start[i]);
    else {
      fft->switchData(size,y+start[i]);
      //   cout<<"Used switchdata"<<endl;
    }
    
    // cout<<"Doing transform"<<endl;
    fft->fwdTransform();  //do the transform
    //  cout<<"Computing power spectrum"<<endl;
    fft->compPowerSpec();

    //do the plotting here
    //    cout<<"Generating the graph"<<endl; 
    if (f==NULL || fft->GetPowerSize()!=oldpowersize ){  //initialize frequencies
      //     cout<<"Initializing arrays"<<endl;
      if (f!=NULL) delete [] f;
      oldpowersize=fft->GetPowerSize();
      f=new Double_t[oldpowersize];
      spec =new Double_t[oldpowersize];
      for(j=0;j<fft->GetPowerSize();j++){f[j]=((double)j)/size;}
      oldsize=size;
      //     cout<<"done"<<endl;
    }
    if(size!=oldsize) for(j=0;j<fft->GetPowerSize();j++){f[j]=((double)j)/size;}
    
    for(j=0;j<fft->GetPowerSize();j++){spec[j]=(fft->GetPowerSpectrum())[j];}
    //above line is a safety measure is fftw_real is not a double


    //    cout<<"The number of points in the graph is "<<fft->GetPowerSize()<<endl;
    graph=new TTuneGraph(fft->GetPowerSize(),f,spec);
    graph->SetName(name);
    graph->SetLineColor(i+1);
    graph->SetMarkerColor(i+1);
    graph->SetMarkerSize(0.7);
    graph->SetMarkerStyle(8); 
    graph->SetMinimum(fft->GetPowerSpectrum()[fft->GetMinPowerBin()]);
    npeaks=graph->Search(1,1e-4);  // search for peaks!
    if(totmin>fft->GetPowerSpectrum()[fft->GetMinPowerBin()])totmin=fft->GetPowerSpectrum()[fft->GetMinPowerBin()];
    if(Nparticles>1){
      mg->Add(graph);
      leg->AddEntry(graph,name,"LP");
    }
    
    
  }
  
  //draw the graphs
  if(Nparticles>1){
    mg->Draw(option);
    mg->SetMinimum(totmin);
    if (!opt.Contains("goff")) leg->Draw();
    xax=mg->GetXaxis();
    yax=mg->GetYaxis();
  }else{
    graph->SetTitle(title);
    graph->Draw(option);
    xax=graph->GetXaxis();
    yax=graph->GetYaxis();
  }

  //adjsut the axes
  if(totmin>0.0 && !opt.Contains("goff")) gPad->SetLogy();  //make the plot log.  
  xax->SetTitle("Tune");
  xax->CenterTitle();
  yax->SetTitle("Amplitude");
  yax->CenterTitle();


  delete fft;
  delete [] start;
  delete histo;
  delete [] f;
  delete [] spec;

  return 1;

}
Int_t CORAANT::TTurnPlot::TuneFootPrint(Option_t* option)
{

  Float_t xmax,ymax;
  TTuneGraph *g;

  TTuneSpace *tunes=new TTuneSpace("Tunes",Form("Tune Footprint for %s",this->GetName()));

// go throught each particle and get the peak tune in each tune spectrum for each plane
  for(int i=0;i<GetParticles();i++){ 
    xmax=ymax=0;   
    TuneGraph("x",Form("N==%i",i),"goff");  //get the x plane and its tune.
    g=(TTuneGraph *)(gROOT->FindObject(Form("Particle:%i",i)));
    xmax=g->GetMaxPeakPositionX();
    delete g;
    TuneGraph("y",Form("N==%i",i),"goff"); //get the y plane and its tune.
    g=(TTuneGraph *)(gROOT->FindObject(Form("Particle:%i",i)));
    ymax=g->GetMaxPeakPositionX(); //the X here is the X axis of g!!!
    delete g;
    if(xmax>=0 && ymax>=0) tunes->Fill(xmax,ymax);
    else Warning("TuneFootPrint","Particle %i does not have good tune data", i);
  }

  tunes->SetMarkerStyle(kFullCircle);
  tunes->SetMarkerSize(1);
  tunes->Draw(option);

  return 1;

}

