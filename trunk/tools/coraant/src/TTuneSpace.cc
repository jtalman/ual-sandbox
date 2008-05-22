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

#include <iostream>
#include <cmath>
#include <algorithm>
#include "TVirtualPad.h"
#include "TTuneSpace.hh"

using namespace std;

ClassImp(CORAANT::TTuneSpace);

inline Int_t sign(double x){ return (x<0) ? -1 : 1;}

CORAANT::TTuneSpace::TTuneSpace():TH2D()
{ 
  order=9;
  GetXaxis()->CenterTitle();
  GetYaxis()->CenterTitle();
  GetXaxis()->SetTitle("Horizontal Tune");
  GetYaxis()->SetTitle("Vertical Tune");
  GetYaxis()->SetTitleOffset(1.5);
  SetLineColor(2);
}

CORAANT::TTuneSpace::TTuneSpace(const char* name, const char* title, Int_t nbins):TH2D(name, title, nbins, 0, 1, nbins, 0, 1)
{
  order=9;
  GetXaxis()->CenterTitle();
  GetYaxis()->CenterTitle();
  GetXaxis()->SetTitle("Horizontal Tune");
  GetYaxis()->SetTitle("Vertical Tune");
  GetYaxis()->SetTitleOffset(1.5);
  SetLineColor(2);
}
   
CORAANT::TTuneSpace::TTuneSpace(const TTuneSpace& h2d):TH2D()
{

  ((TTuneSpace&)h2d).Copy(*this);

}

CORAANT::TTuneSpace::~TTuneSpace(){;}


void CORAANT::TTuneSpace::Copy(TObject& hnew) const 
{

  ((TTuneSpace &)hnew).order=order;
  TH2D::Copy((TH2D&)hnew);

}

void CORAANT::TTuneSpace::Paint(Option_t* option)
{

  int i,k,m,n,kmin,kmax;
  Double_t  nux[2],nuy[2];
  Double_t xmin,xmax,ymin,ymax;
  Double_t edges[4];
  Double_t x,y;
  const Double_t epsilon=1e-15; //used to deal with roundoff and precision errors

  TH2D::Paint(option);
  //gPad->

  TAttLine::Modify();
  xmin=gPad->GetUxmin()-epsilon;
  xmax=gPad->GetUxmax()+epsilon;
  ymin=gPad->GetUymin()-epsilon;
  ymax=gPad->GetUymax()+epsilon;

  //cout<<"THe edges of the historgram are "<<xmin<<" "<<xmax<<" "<<ymin<<" "<<ymax<<endl;

  //thanks to Andy Meyer for this code.
  for(i=2;i<=order;i++){
    for(m=-i;m<=i;m++){
      n=i-abs(m);
      edges[0]=m*xmin+n*ymin;
      edges[1]=m*xmax+n*ymin;
      edges[2]=m*xmin+n*ymax;
      edges[3]=m*xmax+n*ymax;
      kmin= ceil(*min_element(edges,edges+3));
      kmax= floor(*max_element(edges,edges+3))+1;
      // cout<<"i = "<<i<<" m= "<<m<<" n= "<<n<<" kmin="<<kmin<<" kmax ="<<kmax<<endl;      
      // cout<<"THe edges are "<<edges[0]<<" "<<edges[1]<<" "<<edges[2]<<" "<<edges[3]<<endl;
      for(k=kmin;k<kmax;k++){
	nux[0]=nux[1]=nuy[0]=nuy[1]=0.0;
	//cout<<"i = "<<i<<" m= "<<m<<" n= "<<n<<" k= "<<k<<endl;    
	if(n!=0){
	  y=((Double_t)k)/n-m*xmin/n;
	  if(y+2*epsilon>=ymin && y<=ymax){
	    nux[0]=xmin;
	    nuy[0]=y;
	  }
	  y=((Double_t)k)/n-m*xmax/n;
	  // cout<<"ymin= "<<ymin<<" y = "<<y<<" ymax= "<<ymax<<endl;
	  if(y+2*epsilon>=ymin && y<=ymax){
	    nux[1]=xmax;
	    nuy[1]=y;
	  }
	}
	if(m!=0 && n!=m){
	  x=((Double_t)k)/m-n*ymin/m;
	  if(x>=xmin && x<=xmax){
	    nux[0]=x;
	    nuy[0]=ymin;
	  }
	  x=((Double_t)k)/m-n*ymax/m;
	  if(x>=xmin && x<=xmax){
	    nux[1]=x;
	    nuy[1]=ymax;
	  }
	}
	//	cout<<"Testing the line "<<nux[0]<<" "<<nuy[0]<<" "<<nux[1]<<" "<<nuy[1]<<endl;
	if(nux[0]!=0 || nux[1]!=0 || nuy[0]!=0 || nuy[1]!=0){
	  //cout<<"i = "<<i<<" m= "<<m<<" k= "<<k<<" kmin="<<kmin<<" kmax ="<<kmax<<endl;
	  // cout<<"Drawing line "<<nux[0]<<" "<<nuy[0]<<" "<<nux[1]<<" "<<nuy[1]<<endl;
	  gPad->PaintLine(nux[0],nuy[0],nux[1],nuy[1]);
	}
      }//close for k;
    }//close for m
  }//close for i;

  return;
}

      
