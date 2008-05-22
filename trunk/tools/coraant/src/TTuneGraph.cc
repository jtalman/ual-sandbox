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

#include "TVirtualPad.h"
#include "TString.h"
#include "TH1.h"
#include "TSpectrum.h"
#include "TTuneGraph.hh"

#include <iostream>
#include <algorithm>

using namespace std;

ClassImp(CORAANT::TTuneGraph);

void CORAANT::TTuneGraph::SetDefaults()
{

  positionsX=positionsY=NULL;
  npeaks=0;
  peaklines.SetLineColor(2);
  peaklines.SetLineStyle(2);
  peaklines.SetLineWidth(1);

}


CORAANT::TTuneGraph::~TTuneGraph()
{

  if(positionsX) delete [] positionsX;
  if(positionsY) delete [] positionsY;

}


void CORAANT::TTuneGraph::PaintGraph(Int_t npoints, const Double_t* x, const Double_t* y, Option_t* option)
{
  
  //option = 'D'    :The lines denoting peaks are drawn on the graph.
  //for other options, see TGraph::PaintGraph

  int i;
  TString opt = option;
  opt.ToUpper();

 
  TGraph::PaintGraph(npoints, x, y, option);
  
  if(opt.Contains("D") && npeaks>0){  //draw lines at each point
    peaklines.Modify();
    Float_t max=gPad->GetUymax();
    Float_t min=gPad->GetUymin();
    for(i=0;i<npeaks;i++) gPad->PaintLine(positionsX[i], min, positionsX[i], max);
  }

}


Int_t CORAANT::TTuneGraph::Search(Double_t sigma, Double_t threshold, Option_t *option)
{
  //  The peak finder is the TSpectrum class, see class description for meaning of the
  //  sigma and threshold.
  //  If the peak is close to a minimum, the parabolic interpolation can sometimes lead to
  //  the minimum.
  

  Int_t size=GetN(); 
  Int_t i,j,pos,inc=0;
  Float_t *dest   = new Float_t[size];
  Float_t *source = new Float_t[size];
  Float_t x[3],y[3];  //used for parabolic fitting
  Float_t X;  //just used to make the source look cleaner!
  TSpectrum spec;
  bool leftside, rightside;

  for(i=0;i<size;i++){source[i]=GetY()[i];}
  
  spec.Search1HighRes(source, dest, size, sigma, 100*threshold, kTRUE, 3, kTRUE, 3); 

  if(npeaks!=spec.GetNPeaks()){ 
    npeaks=spec.GetNPeaks();
    if (positionsX) delete [] positionsX;
    if (positionsY) delete [] positionsY;
    if(npeaks!=0){
      positionsX=new Float_t [npeaks];
      positionsY=new Float_t [npeaks];
    }else positionsX=positionsY=NULL;
  }
			    
  for (i = 0; i < npeaks; i++) {
    pos=spec.GetPositionX()[i];  //get the index of the peak
    inc=0;
    do{
      for(j=0;j<3;j++){  //get the three pointd around the peak
	x[j]=GetX()[pos+inc-1+j];
	y[j]=GetY()[pos+inc-1+j];
      }     
      leftside= y[0]<=y[1] && y[1]<=y[2];
      rightside= y[0]>=y[1] && y[1]>=y[2];
      if(leftside) inc++; //if sitting on the left side of peak
      if(rightside) inc--; //if sitting on the right side of peak
    }while((leftside || rightside)); //loop while sitting on the side of peak

    //evaluate the X and Y of the maximum
    positionsX[i]=x[1]-0.5*((x[1]-x[0])*(x[1]-x[0])*(y[1]-y[2])-(x[1]-x[2])*(x[1]-x[2])*(y[1]-y[0]))/
      ((x[1]-x[0])*(y[1]-y[2])-(x[1]-x[2])*(y[1]-y[0]));
    X=positionsX[i];
    positionsY[i]=y[0]*(X-x[1])*(X-x[2])/((x[0]-x[1])*(x[0]-x[2])) + y[1]*(X-x[0])*(X-x[2])/((x[1]-x[0])*(x[1]-x[2]))
      +y[2]*(X-x[1])*(X-x[0])/((x[2]-x[0])*(x[2]-x[1]));

    //    if(y[0]>=y[1] && y[1]<=y[2]) cout<<"Peak "<<i<<"is a minimum!"<<endl;
   
  }


  delete [] dest;
  delete [] source;
  

  return npeaks;
}


Float_t CORAANT::TTuneGraph::GetMaxPeakPositionX()
{

  int index=0;
  Float_t maximum=-1;
  Float_t newmax;
 
  if (npeaks==0){
    cerr<<"CORAANT::TTuneGraph::GetMaxPeakPositionX - A search for peaks was never done or none where found"<<endl;
    return -1;
  }

  for(int i=0;i<npeaks;i++){
    if (positionsY[i]>0 && positionsY[i]<1e300){
      newmax=max(maximum,positionsY[i]);
      if(maximum!=newmax){
	index=i;
	maximum=newmax;
      }
    }
  }

    //  index=max_element(positionsY,positionsY+npeaks)-positionsY;

  return positionsX[index];
}      

Float_t CORAANT::TTuneGraph::GetMaxPeakPositionY()
{  

  int index;
  Float_t maximum=-1;
  Float_t newmax;

  if (npeaks==0){
    cerr<<"CORAANT::TTuneGraph::GetMaxPeakPositionX - A search for peaks was never done or none where found"<<endl;
    return -1;
  }

  for(int i=0;i<npeaks;i++){
    if (positionsY[i]>0 && positionsY[i]<1e300){
      newmax=max(maximum,positionsY[i]);
      if(maximum!=newmax) maximum=newmax;     
    }
  }

    //  index=max_element(positionsY,positionsY+npeaks)-positionsY;

  return maximum;
}      

