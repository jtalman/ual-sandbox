/* ****************************************************************************
   *                                                                          *
   *   This C++ code is an example of how to develop a ROOT-based shell to    *
   *   UAL simulation environment class can be                                *
   *                                                                          *
   *   To use the IO capabilities of ROOT, a class must:                      *
   *   1) Ultimately inherit from TObject.                                    *
   *   2) Use the macro ClassDef(classname,version) in the header file        *
   *   3) Use the macro ClassImp(classname) in the .cc file                   *
   *   4) the makefile must generate the dictionary/streamer file             *
   *   5) The class dictionary file must be linked to the class file          *
   *                                                                          *
   *   The ClassDef and ClassImp macros define other class members that       *
   *   are needed to use ROOT IO and RTTI facilities.  Classes that do        *
   *   no use these facilities do not need these modifications                *
   *                                                                          *
   *                                                                          *
   *   Author: Ray Fliller III and Nikolay Malitsky                           *
   *                                                                          *
   *                                                                          *
   **************************************************************************** */

#include <cstring>
#include <iostream>
#include <iomanip>
#include <fstream>
#include "timer.h"

//UAL Libraries
#include "UAL/APDF/APDF_Builder.hh"
#include "UAL/APF/PropagatorSequence.hh"
#include "UAL/APF/PropagatorNodePtr.hh"
#include "UAL/SXF/Parser.hh"
//#include "Optics/PacTwissData.h"
#include "Templates/PacVector.h"
#include "TEAPOT/Integrator/LostCollector.hh"
#include "Main/TeapotTwissService.h"

// ROOT libraries
#include "TMath.h"
#include "TNtupleD.h"
#include "TRandom.h"

#include "RootShell.hh"

#define OVECCOUNT 30    /* should be a multiple of 3 */


ClassImp(UAL::RootShell); 

using namespace std;

ZLIB::Space UAL::RootShell::s_space(6, 5);
R__EXTERN TRandom *gRandom;

UAL::RootShell::RootShell()
{
  m_ap=NULL;
  m_tea=NULL;
  m_twiss=NULL;
  m_orbit.set(0,0,0,0,0,0);

}

UAL::RootShell::~RootShell()
{ 
  if(m_tea!=NULL) delete m_tea;
  if(m_twiss!=NULL) delete [] m_twiss;
  if(m_ap!=NULL) delete m_ap;
} 


void UAL::RootShell::readSXF(const char* inFile, const char* outDir)
{
  string echoFile = outDir; echoFile += "/sxf.echo";
  string outFile =  outDir; outFile  += "/sxf.out";

  UAL::SXFParser parser;
  parser.read(inFile, echoFile.data()); 
  parser.write(outFile.data());
}

void UAL::RootShell::use(const char* latticeName)
{
  string accName = latticeName;

  PacLattices::iterator latIterator = PacLattices::instance()->find(accName);
  if(latIterator == PacLattices::instance()->end()){
     cerr << "There is no " + accName << " accelerator " << endl;
    exit(1);
  }

  m_lattice = *latIterator;

  if (m_tea!=NULL) delete m_tea;
  if (m_twiss!=NULL) delete [] m_twiss;
  m_tea=new Teapot(m_lattice);

  /*
  cout<<"We are using lattice "<<m_lattice.getName()<<endl;
  cout<<"THere are "<< m_lattice.getNodeCount()<<" elements"<<endl;

  for(i=0;i<m_lattice.getNodeCount();i++){
     if(m_lattice.getNodeAt(i)->getType()=="Rcollimator"){
      cout<<"THe index of the collimator is "<<i<<endl;
    }
  }
  */

}

void UAL::RootShell::setBeamAttributes(const PAC::BeamAttributes& ba)
{
  m_ba = ba;
}

void UAL::RootShell::setBeamAttributes(const double energy, const double mass, const double charge)
{
  m_ba.setEnergy(energy);
  m_ba.setMass(mass);
  m_ba.setCharge(charge);
}

void UAL::RootShell::generateBunch(PAC::Bunch& bunch, int at)
{

  Double_t pos[6];
  PacTwissData Twiss;
  //  Teapot Tea(m_lattice);
  int i;
  Double_t Jx,Jy,delta,phiX,phiY;

  double gamma = m_ba.getEnergy()/m_ba.getMass();
  Double_t betagamma=sqrt(gamma*gamma -1);
  

  // bunch.resize(Nparticles); 
  bunch.getBeamAttributes().setEnergy(m_ba.getEnergy());
  bunch.getBeamAttributes().setMass(m_ba.getMass());
  bunch.getBeamAttributes().setCharge(m_ba.getCharge());

  //get Twiss parameters for location index at
  Twiss=getTwiss(at);
  
  //generate bunch
  for(i = 0; i < bunch.size(); i++){
    //pick distribution, exponential in action, uniform in phase, gaussian in longitudinal
    phiX=gRandom->Uniform(0,2*3.14159);
    phiY=gRandom->Uniform(0,2*3.14159);
    delta=gRandom->Gaus(0,1e-3);
    Jx=gRandom->Exp(15e-6/(6*betagamma));
    Jy=gRandom->Exp(15e-6/(6*betagamma));

    //match transverse to phase space
    pos[0]=m_orbit[0]+sqrt(2*Jx*Twiss.beta(0))*cos(phiX)+Twiss.d(0)*delta;
    pos[1]=m_orbit[1]-sqrt(2*Jx/Twiss.beta(0))*(sin(phiX)+Twiss.alpha(0)*cos(phiX))+Twiss.dp(0)*delta;
    pos[2]=m_orbit[2]+sqrt(2*Jy*Twiss.beta(1))*cos(phiY)+Twiss.d(1)*delta;
    pos[3]=m_orbit[3]-sqrt(2*Jy/Twiss.beta(1))*(sin(phiY)+Twiss.alpha(1)*cos(phiY))+Twiss.dp(1)*delta;
    pos[4]=m_orbit[4]+gRandom->Gaus(0,1e-6);
    pos[5]=m_orbit[5]+delta;

    //add the particle to the bunch  
    bunch[i].getPosition().set(pos[0],pos[1],pos[2],pos[3],pos[4],pos[5]);    
  }
 
  // return bunch;
}


void UAL::RootShell::readAPDF(const char* inFile)
{
  UAL::APDF_Builder apBuilder; 
  apBuilder.setBeamAttributes(m_ba);

  string apdfFile(inFile);
  m_ap = apBuilder.parse(apdfFile);
  if(m_ap == 0) {
     cout << "RootShell::readAPDF - Accelerator Propagator has not been created " <<  endl;
  }
}

PacTwissData UAL::RootShell::getTwiss(int index)
{

  int size=m_ap->getRootNode().size();

  if(m_tea==NULL){
    cerr<<"RootShell::getTwiss - select a lattice to use first "<<endl;
    return *m_twiss;
  }

  if(index>=size){
    cerr<<"RootShell::getTwiss - The number of elements in the lattice is "<<size<<endl;
    return *m_twiss;
  }

  if(m_twiss==NULL){
    m_tea->clorbit(m_orbit,m_ba);
    TeapotTwissService service(*m_tea);
    m_twiss = new PacTwissData[size];
    PacVector<int> indices(size);
    for(int i=0; i< size; i++){ indices[i] = i;}
    service.define(m_twiss, indices, m_ba, m_orbit);
  }
  
  
  return m_twiss[index];
}

void UAL::RootShell::track(PAC::Bunch &bunch, int start, int end)
{
  // If end<=start, it is assumed that the tracking will loop RHIC past the 
  // end of RHIC


  int i;
  int size=m_ap->getRootNode().size();
  

  cout << "\nAPF-based Teapot tracker " << endl;  
  cout << "size : " <<size << " propagators " << endl;
  if (start>=size) {
    cerr<<"RootShell::track - Cannot start at "<<start<<" it is past the end"<<endl;
    return;
  }

  if(end==-1 || start==0) end=size-1;
  if(end>=size){
    cerr<<"RootShell::track - End is too large.  Ending at last element"<<endl;
    end=size-1;
  }

  PropagatorSequence RootNode=m_ap->getRootNode();
  
  list<UAL::PropagatorNodePtr>::iterator begin = RootNode.begin();
  list<UAL::PropagatorNodePtr>::iterator finish= RootNode.begin();  
  list<UAL::PropagatorNodePtr>::iterator it;

  for(i=0;i<size;i++){  // set the pointers to the correct nodes
    if(i<start) begin++;
    if(i<end) {
      finish++;
      //      it++;
    }
    if(i >start && i>end) break;
  }
  finish++;  //to go one past the end;
  

  //  cout<<"THe first node is "<<(*begin)->getFrontAcceleratorNode().getName()<<endl;
  //cout<<"THe last node is "<<(*it)->getFrontAcceleratorNode().getName()<<endl;

  TEAPOT::LostCollector::GetInstance().SetTurn(0);
  if(end>start){  //do not have to worry about end of ring
    //  cout<<"Tracking from "<< (*begin)->getFrontAcceleratorNode().getName()<<" to "<<(*it)->getFrontAcceleratorNode().getName()<<endl;
    for(it = begin; it != finish; it++)(*it)->propagate(bunch);    
  }
  else{
    // cout<<"Tracking from "<< (*begin)->getFrontAcceleratorNode().getName()<<endl;
    for(it = begin; it != RootNode.end(); it++)(*it)->propagate(bunch);
    // cout<<"Looping around the end of the ring"<<endl;
    //cout<<"Tracking to element before"<< (*finish)->getFrontAcceleratorNode().getName()<<endl;
    for(it = RootNode.begin(); it != finish; it++)(*it)->propagate(bunch);
  }


}

void UAL::RootShell::multitrack(PAC::Bunch &bunch, int Nturns, int start)
{
  


  int i;
  int size=m_ap->getRootNode().size();


  cout << "\nAPF-based Teapot tracker " << endl;  
  cout << "size : " <<size << " propagators " << endl;
  if (start>=size) {
    cerr<<"RootShell::multitrack - Cannot start at "<<start<<" it is past the end"<<endl;
    return;
  }
  cout <<" Starting at element "<<start<<endl;
  cout<<"Tracking for "<<Nturns<<" turns"<<endl;
  
  PropagatorSequence RootNode=m_ap->getRootNode();
  
  list<UAL::PropagatorNodePtr>::iterator begin = RootNode.begin();
  list<UAL::PropagatorNodePtr>::iterator it;
  
  if(start!=0){
    for(i=0;i<start;i++) begin++;
  }

  for(i=0;i<Nturns;i++){
    TEAPOT::LostCollector::GetInstance().SetTurn(i);
    cout<<"Tracking turn "<<i<<" of "<<Nturns<<endl;
    if(begin==RootNode.begin())  m_ap->propagate(bunch);  //starting at the beginning
    else{
      //cout<<"tracking element by element starting at "<<start<<" to the end "<<size<<endl;
      for(it = begin; it != RootNode.end(); it++)(*it)->propagate(bunch); //from start to end of ring
      //cout<<"tracking element by element from start to "<<start<<endl;
      for(it = RootNode.begin(); it != begin; it++)(*it)->propagate(bunch); //from beginning of ring to start

    }
  }

}


UAL::AcceleratorNode* UAL::RootShell::GetLattice()
{
  return &m_lattice;
}

void UAL::RootShell::steer(const char *adjusters, const char *detectors, const char plane)
{

  PacVector<int> ads;
  PacVector<int> dets;
  
  if(!strcmp(adjusters,"")){
    cerr<<"RootShell::steer - Cannot adjust steering in "<<plane<<" plane:  No correctors specified"<<endl;
    return;
  }
  if(!strcmp(detectors,"")){
    cerr<<"RootShell::steer - Cannot adjust steering in "<<plane<<" plane:  No detectors specified"<<endl;
    return;
  }


  ads=m_lattice.indexes(adjusters);
  dets=m_lattice.indexes(detectors);

  if(ads.size()==0){
    cerr<<"RootShell::steer - There are no adjusters. Cannot flatten orbit."<<endl;
    return;
  }
  if(dets.size()==0){
    cerr<<"RootShell::steer - There are no detectors. Cannot flatten orbit."<<endl;
    return;
  }
  

  // do the correction
 
  m_tea->steer(m_orbit, m_ba, ads,dets, 1, plane);
  

}

void UAL::RootShell::tunethin(const char *focus, const char *defocus, double mux, double muy, char method,
		  int numtries, double tolerance, double stepsize)
{

  PacVector<int> foc;
  PacVector<int> def;

   if(!strcmp(focus,"")){
     cerr<<"RootShell::tunethin - Cannot tune lattice, no focussing quads specified"<<endl;
     return;
  }
  if(!strcmp(defocus,"")){
    cerr<<"RootShell::tunethin - Cannot tune lattice, no defocussing quads specified"<<endl;
    return;
  }

  foc=m_lattice.indexes(focus);
  def=m_lattice.indexes(defocus);

  if(foc.size()==0){
    cerr<<"RootShell::tunethin - There are no focussing quads. Cannot tune lattice."<<endl;
    return;
  }
  if(def.size()==0){
    cerr<<"RootShell::tunethin - There are no defocussing quads. Cannot tune lattice."<<endl;
    return;
  }

  m_tea->tunethin(m_ba, m_orbit, foc, def, mux, muy, method, numtries, tolerance, stepsize);
  
}

void UAL::RootShell::chromfit(const char *focus, const char *defocus, double xix, double xiy, char method,
 		      int numtries, double tolerance, double stepsize)
{

  PacVector<int> foc;
  PacVector<int> def;
  
  if(!strcmp(focus,"")){
    cerr<<"RootShell::chromfit - Cannot tune lattice, no focussing sextupoles specified"<<endl;
    return;
  }
  if(!strcmp(defocus,"")){
    cerr<<"RootShell::chromfit - Cannot tune lattice, no defocussing sextupoles specified"<<endl;
    return;
  }
  
  foc=m_lattice.indexes(focus);
  def=m_lattice.indexes(defocus);
  
  if(foc.size()==0){
    cerr<<"RootShell::chromfit - There are no focussing sextupoles. Cannot tune lattice."<<endl;
    return;
  }
  if(def.size()==0){
    cerr<<"RootShell::chromfit - There are no defocussing sextupoles. Cannot tune lattice."<<endl;
    return;
  }
  
  m_tea->chromfit(m_ba, m_orbit, foc, def, xix, xiy, method, numtries, tolerance, stepsize);
  
}

void UAL::RootShell::decouple(const char *a11, const char *a12, const char *a13, const char *a14, 
 		  const char *focus, const char *defocus, double mux, double muy)
{

  PacVector<int> foc;
  PacVector<int> def;
  PacVector<int> a11vec;
  PacVector<int> a12vec;
  PacVector<int> a13vec;
  PacVector<int> a14vec;
  
  if(!strcmp(focus,"")){
    cerr<<"RootShell::decouple - Cannot tune lattice, no focussing quads specified"<<endl;
    return;
  }

  if(!strcmp(defocus,"")){
    cerr<<"RootShell::decouple - Cannot tune lattice, no defocussing quads specified"<<endl;
    return;
  }

  if(!strcmp(a11,"")){
    cerr<<"RootShell::decouple - Cannot tune lattice, no a11 magnets specified"<<endl;
    return;
  }

  if(!strcmp(a12,"")){
    cerr<<"RootShell::decouple - Cannot tune lattice, no a12 magnets specified"<<endl;
    return;
  }
  
  if(!strcmp(a13,"")){
    cerr<<"RootShell::decouple - Cannot tune lattice, no a13 magnets specified"<<endl;
    return;
  }
  
  if(!strcmp(a14,"")){
    cerr<<"RootShell::decouple - Cannot tune lattice, no a14 magnets specified"<<endl;
    return;
  }

  a11vec=m_lattice.indexes(a11);
  a12vec=m_lattice.indexes(a12);
  a13vec=m_lattice.indexes(a13);
  a14vec=m_lattice.indexes(a14);
  foc=m_lattice.indexes(focus);
  def=m_lattice.indexes(defocus);
  
  if(foc.size()==0){
    cerr<<"RootShell::decouple - There are no focussing quads. Cannot tune lattice."<<endl;
    return;
  }
  if(def.size()==0){
    cerr<<"RootShell::decouple - There are no defocussing quads. Cannot tune lattice."<<endl;
    return;
  }
  if(a11vec.size()==0){
    cerr<<"RootShell::decouple - There are no a11 magnets. Cannot tune lattice."<<endl;
    return;
  }
  if(a12vec.size()==0){
    cerr<<"RootShell::decouple - There are no a12 magnets. Cannot tune lattice."<<endl;
    return;
  }
  if(a13vec.size()==0){
    cerr<<"RootShell::decouple - There are no a13 magnets. Cannot tune lattice."<<endl;
    return;
  }
  if(a14vec.size()==0){
    cerr<<"RootShell::decouple - There are no a14 magnets. Cannot tune lattice."<<endl;
    return;
  }

  m_tea->decouple(m_ba, m_orbit, a11vec, a12vec, a13vec, a14vec, foc, def, mux, muy);
 
}

//     void UAL::RootShell::map(int order, const char *filename);
//     void UAL::RootShell::matrix(double delta, const char *filename);


void UAL::RootShell::analysis(const char *filename, double delta)
{

  PacChromData chrom;
  PacTwissData twiss;

  ofstream file(filename,ios::ate); //open and append to end

  double v0=sqrt(m_ba.getEnergy()*m_ba.getEnergy()-m_ba.getMass()*m_ba.getMass())/m_ba.getEnergy();
  double twopi=2.0*TMath::Pi();
  
  if(!file.is_open()){
    cerr<<"Cannot open file "<<filename<<endl;
    return;
  }

  if (delta!=0) m_orbit.setDE(0);

  cout<<"Finding closed orbit "<<endl;
  file.fill(' ');
  file<<endl;
  file<<setw(15)<<"x[m]"<<setw(15)<<"px"<<setw(15)<<"y[m]"<<setw(15)<<"py"<<setw(15)<<"dE/p"<<endl;
 
  m_tea->clorbit(m_orbit,m_ba);
  
  file.precision(10);
  file<<setw(14)<<m_orbit[0]<<" "<<setw(14)<<m_orbit[1]<<" "<<setw(14)<<m_orbit[2]<<" "<<setw(14)<<m_orbit[3]<<" "<<setw(14)<<m_orbit[5]<<endl<<endl;
 
  cout<<"Finding Twiss Parameters"<<endl;
  m_tea->chrom(chrom,m_ba,m_orbit);
  
  twiss=chrom.twiss();

  file<<"Twiss Parameters"<<endl;
  file<<setw(15)<<"beta"<<setw(15)<<"alpha"<<setw(15)<<"q"<<setw(15)<<"d(dp/p)"<<setw(15)<<"dd(dp/p)"<<setw(15)<<"dq(dp/p)"<<endl; 
  
  file.precision(9);
  
  file<<setw(14)<<twiss.beta(0)<<" " <<setw(14)<< twiss.alpha(0)<<" " <<setw(14)<< twiss.mu(0)*twopi<<" " <<setw(14)<< v0*twiss.d(0)<<" " <<setw(14)<< v0*v0*twiss.dp(0)<<" ";
  file<<setw(14)<< v0*chrom.dmu(0)*twopi <<endl;
  file<<setw(14)<<twiss.beta(1)<<" " <<setw(14)<< twiss.alpha(1)<<" " <<setw(14)<< twiss.mu(1)*twopi<<" " <<setw(14)<< v0*twiss.d(1)<<" " <<setw(14)<< v0*v0*twiss.dp(1)<<" ";
  file<<setw(14)<< v0*chrom.dmu(1)*twopi <<endl;

  file.close();
  
  cout<<"Done."<<endl;
}
  
						         
  

  
  
