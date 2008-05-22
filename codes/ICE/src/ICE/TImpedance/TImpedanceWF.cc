
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <math.h>
#include "UAL/Common/Def.hh"
#include "ICE/TImpedance/TImpedanceWF.hh"

// Constructor

ICE::TImpedanceWF::TImpedanceWF(int nBins, int nImpElements, int maxBunchSize)
  : coefficient_(0.0) , time_min_prev_(0.0)
{
  int i,j;

  nBins_   = nBins;

  maxBunchSize_ = maxBunchSize;

  nElem_   = nImpElements;

  xyM_     = new double* [2];
  xyM_[0]  = new double[nBins_ + 1]; // x-coordinate
  xyM_[1]  = new double[nBins_ + 1]; // y-coordinate

  for(j = 0; j <= nBins_; j++) {
    xyM_[0][j] = 0.0;
    xyM_[1][j] = 0.0;
  }

  tSumKick_    = new double* [2]; 
  tSumKick_[0] = new double[nBins_ + 1];
  tSumKick_[1] = new double[nBins_ + 1];

    for(i = 0; i <= nBins_; i++){
     tSumKick_ [0][i] = 0.0;
     tSumKick_ [1][i] = 0.0;
    }

  xy_mask_ = new int [2];
  xy_mask_[0] = 0; // don't calulate x-axis kick
  xy_mask_[1] = 0; // don't calulate y-axis kick

  tKick_    = new ICE::CmplxVector** [2];
  tKick_[0] = new ICE::CmplxVector* [nBins_ + 1]; // x-axis kick array
  tKick_[1] = new ICE::CmplxVector* [nBins_ + 1]; // y-axis kick array

    for(i = 0; i <= nBins_; i++){
     tKick_ [0][i]  = new ICE::CmplxVector(nElem_);
     tKick_ [1][i]  = new ICE::CmplxVector(nElem_);
    }

  //current part of kick from previous bunches and "memory" for kick

  tfKick_   =  new ICE::CmplxVector* [2];
  tfKick_[0]=  new ICE::CmplxVector(nElem_); 
  tfKick_[1]=  new ICE::CmplxVector(nElem_);

  tFFKick_   =  new ICE::CmplxVector* [2];
  tFFKick_[0]=  new ICE::CmplxVector(nElem_); 
  tFFKick_[1]=  new ICE::CmplxVector(nElem_);    

  //additional memory for to store the tFFKick_ memory
  tFFmemory_   =  new ICE::CmplxVector* [2];
  tFFmemory_[0]=  new ICE::CmplxVector(nElem_); 
  tFFmemory_[1]=  new ICE::CmplxVector(nElem_); 

  // Characteristics of Wake Fileds 
  //    wake field at zero distance [Ohms/(m^2)] and 
  //    eta-coff [1/sec] for shifting wake fields

  wake_zero_   =  new ICE::CmplxVector* [2];
  wake_zero_[0]=  new ICE::CmplxVector(nElem_); 
  wake_zero_[1]=  new ICE::CmplxVector(nElem_);

  eta_coff_   =  new ICE::CmplxVector* [2];
  eta_coff_[0]=  new ICE::CmplxVector(nElem_); 
  eta_coff_[1]=  new ICE::CmplxVector(nElem_);

  eta_fact_   =  new ICE::CmplxVector* [2];
  eta_fact_[0]=  new ICE::CmplxVector(nElem_); 
  eta_fact_[1]=  new ICE::CmplxVector(nElem_);
  
  // number of actual elements
  nElemExist_ = new int [2];
  nElemExist_[0] = 0;
  nElemExist_[1] = 0;

  // bins_[ip] is the index of the bin position as function 
  //           of the  macro-particle index.
 
  bins_         = new int[maxBunchSize_];

  fractBins_    = new double[maxBunchSize_]; 

  for(j = 0; j < maxBunchSize_; j++) {
    bins_[j]      = 0;
    fractBins_[j] = 0.0;
  }
}

// Destructor

ICE::TImpedanceWF::~TImpedanceWF()
{
  int i;

  delete [] xyM_[0];
  delete [] xyM_[1];
  delete [] xyM_;

  delete [] tSumKick_[0];
  delete [] tSumKick_[1];
  delete [] tSumKick_;

  delete [] xy_mask_;

    for(i = 0; i <= nBins_; i++){
     delete tKick_ [0][i];
     delete tKick_ [1][i];
    }
  delete [] tKick_[0];
  delete [] tKick_[1];


  delete tfKick_[0];
  delete tfKick_[1];
  delete [] tfKick_;

  delete tFFKick_[0];
  delete tFFKick_[1];
  delete [] tFFKick_;

  delete tFFmemory_[0];
  delete tFFmemory_[1];
  delete [] tFFmemory_;

  delete wake_zero_[0];
  delete wake_zero_[1];
  delete [] wake_zero_;

  delete eta_coff_[0];
  delete eta_coff_[1];
  delete [] eta_coff_;

  delete eta_fact_[0];
  delete eta_fact_[1];
  delete [] eta_fact_;


  delete [] nElemExist_;


  delete [] bins_;
  delete [] fractBins_;

  //cerr << "Destructor was done. TImpedanceWF  !!!! \n";

}


// Set the parameters of the impedance element's wake field
void ICE::TImpedanceWF::addElement( int i_xy,
				    double wake_zero_re, 
				    double wake_zero_im,
				    double eta_coff_re,
				    double eta_coff_im )
{
  int i;
  i = nElemExist_[i_xy];
  if(i != nElem_ ){
     wake_zero_[i_xy]->setRe(i, wake_zero_re);
     wake_zero_[i_xy]->setIm(i, wake_zero_im);
     eta_coff_[i_xy]->setRe(i, eta_coff_re);
     eta_coff_[i_xy]->setIm(i, eta_coff_im);
  }
  else
    {
      	std::cerr << "You try to add more Generic Transverse Impedance Elements \n";
        std::cerr << "    than the container ICE::TImpedanceWF  has. Stop.  \n";
	// _finalize_MPI();
	exit(1);
    }
  nElemExist_[i_xy]++;
  xy_mask_[i_xy]=1;
  setRange_();
}

// Print the parameters of the impedance element's wake fields
void ICE::TImpedanceWF::printElementParameters()
{
  int i_xy, xy_mask, r_min, r_max;
  int i;
  char s[120];

  std::cerr << " ------Parameters of the impedance element's wake fields-----start-----\n";

  for( i_xy = 0 ; i_xy <= 1 ; i_xy++){
    xy_mask = xy_mask_[i_xy];
    if(xy_mask != 0 ) {
      r_min =  0;
      r_max =  nElemExist_[i_xy]-1;
      if(i_xy == 0 ) { std::cerr << " ----------x-dimension parameters----------\n";}
      if(i_xy == 1 ) { std::cerr << " ----------y-dimension parameters----------\n";}
      for( i=r_min; i<=r_max; i++){
       sprintf(s, " %-2d W(0)(wake field  ) Re Im : %-15.8e  %-15.8e", 
	       i, wake_zero_[i_xy]->getRe(i), wake_zero_[i_xy]->getIm(i));       
       std::cerr << s << "\n";
       sprintf(s, " %-2d Eta (shift vector) Re Im : %-15.8e  %-15.8e", 
	       i, eta_coff_[i_xy]->getRe(i), eta_coff_[i_xy]->getIm(i));       
       std::cerr << s << "\n";
      } 
    }
  }
  std::cerr << " ------Parameters of the impedance element's wake fields-----stop------\n";
}

// Set the parameters of the resonant impedance element's wake field
//     r  - resistance parameter [ohms/m^2]
//     q  - quality factor
//     fr - resonant frequency [Hz]
void ICE::TImpedanceWF::addResonantElement( int i_xy,
					    double r, 
					    double q,
					    double fr)
{
  double f,f_res,z;
  z=1.0/(4*q*q);
  if(z >= 1.0 ){
    std::cerr << "Quality factor is too small. It is not good resonant element. \n";
    // _finalize_MPI();
    exit(1);
  }
  
  f_res = 2*UAL::pi*fr;
  f = f_res*sqrt(1.0 - z); 

  double w0_re,w0_im,eta_re,eta_im;

  w0_re  = -r*f_res/(q*f);
  w0_im  = 0.0;
  eta_re = -f_res/(2.*q);
  eta_im = f;

  int i;
  i = nElemExist_[i_xy];
  if(i != nElem_ ){
     wake_zero_[i_xy]->setRe(i, w0_re);
     wake_zero_[i_xy]->setIm(i, w0_im);
     eta_coff_[i_xy]->setRe(i, eta_re);
     eta_coff_[i_xy]->setIm(i, eta_im);
  }
  else
    {
      	std::cerr << "You try to add more Generic Transvese Resonant Impedance Elements \n";
        std::cerr << "    than the container ICE::TImpedanceWF  has. Stop.  \n";
	// _finalize_MPI();
	exit(1);
    }
  nElemExist_[i_xy]++;
  xy_mask_[i_xy]=1;
  setRange_();
}


//Set the range of operation for CmplxVector's arrays.
//There should be placed all CmplxVector variables
void ICE::TImpedanceWF::setRange_()
{
  int i, j, xy_mask, r_min, r_max;
  for( i = 0 ; i <= 1 ; i++){
    xy_mask = xy_mask_[i];
    if(xy_mask != 0 ) {
      r_min =  0;
      r_max =  nElemExist_[i]-1;
      tfKick_[i]->setRange(r_min, r_max);
      tFFKick_[i]->setRange(r_min, r_max);
      tFFmemory_[i]->setRange(r_min, r_max);      
      wake_zero_[i]->setRange(r_min, r_max);      
      eta_fact_[i]->setRange(r_min, r_max);      
      eta_coff_[i]->setRange(r_min, r_max);
       for(j = 0; j <= nBins_; j++){
        tKick_[i][j]->setRange(r_min, r_max);
       }
    }
  }
}

//Get wake function
double ICE::TImpedanceWF::getWF(int i_xy, int j, double t)
{

  if(i_xy < 0 || i_xy > 1) {
   std::cerr << "ICE::TImpedanceWF::getWF : there are only x and y coordinates (0 or 1 indexes)  \n";
   // _finalize_MPI();
   exit(1);
  }
  int nEl;
  nEl = nElemExist_[i_xy]-1; 
  if(j < 0 || j > nEl) {
    std::cerr << "ICE::TImpedanceWF::getWF : There is not WF-elements with index =" << j <<"\n";
    // _finalize_MPI();
    exit(1);
  }
  
  tfKick_[i_xy]->copy( *wake_zero_[i_xy]);
  tfKick_[i_xy]->shift( *eta_coff_[i_xy], t);
  double w;
  w = tfKick_[i_xy]->getIm(j);
  return w;

}



// Bin the macro particles longitudinally
void ICE::TImpedanceWF::defineLongBin(const PAC::Bunch& bunch)
{
  // Define t_min_ and t_max_ 
  double ct, ct_min, ct_max;
  bunchSize_ = 0;
  ct_min =  1.0e+21;
  ct_max = -1.0e+21;
  int ip;
  for(ip = 0; ip < bunch.size(); ip++){
    if(!bunch[ip].isLost()) {
      bunchSize_++;  
      ct = bunch[ip].getPosition().getCT();
      if( ct > ct_max ) ct_max = ct;   
      if( ct < ct_min ) ct_min = ct;
    }
  }

  if(bunchSize_ == 0 ) {return;}

   if( ct_min == ct_max ) {
    std::cerr << "ICE::TImpedanceWF::defineLongBin : There is zero longitudinal spread in the bunch. Stop.\n";
    // _finalize_MPI();
    exit(1);
   }

   _grid_synchronize( ct_min, ct_max);

  double ct_step;
  ct_step = 1.00000001*(ct_max - ct_min)/nBins_;

  double beta, gamma;
  beta = getBeta_(bunch);
  gamma = 1./sqrt(1. - beta*beta);
  
  coefficient_  = -4*UAL::pi*ICE::pradius/(ICE::Z0*beta*beta*gamma);
  coefficient_ *=  bunch.getBeamAttributes().getMacrosize()*bunch.getBeamAttributes().getCharge();

  t_min_ = ct_min/(beta*UAL::clight); 
  t_max_ = ct_max/(beta*UAL::clight); 

  t_step_ = 1.00000001*(t_max_ - t_min_)/nBins_; 

  for(int i = 0 ; i <= nBins_ ; i++){
   xyM_[0][i] = 0.0;
   xyM_[1][i] = 0.0;
  }

  // binning on the base of variable ct_step
  int iT;
  double  x , y;
  double tt, fract, fract1;
  for(ip = 0; ip < bunch.size(); ip++){
    if(!bunch[ip].isLost()) {
      ct = bunch[ip].getPosition().getCT();
      tt = (ct - ct_min)/ct_step;
      iT = int (tt);
      bins_[ip] = iT;
      fract = tt - iT;
      fract1 = 1.0 - fract;
      fractBins_[ip] = fract;
     
      x  = bunch[ip].getPosition().getX();
      // xyM_[0][iT]   += x;     
      xyM_[0][iT]   += fract1*x;
      xyM_[0][iT+1] += fract*x;
 
      y  = bunch[ip].getPosition().getY();
      // xyM_[1][iT]   += y;
      xyM_[1][iT]   += fract1*y;
      xyM_[1][iT+1] += fract*y;            
    }
  }

  //for parallel version (empty method here)
  _sum_distribution();

}

// Print out the <x> and <y> momenta of bunch
void ICE::TImpedanceWF::showXY(char* f)
{
  std::ofstream file;
  file.open(f);
  if(!file) {
    std::cerr << "ICE::TImpedanceWF::showXY: Cannot open " << f << " for output \n";
    // _finalize_MPI();
    exit(1);
  }
 char s[120];
  int i;
  file << "nBins t_min t_max t_step [sec]:" << nBins_ 
       << "  " << t_min_ 
       << "  " << t_max_
       << "  " << t_step_ 
       << "\n"; 
  file << "i            <x> [m]           <y> [m]  \n";

  for (i = 0; i <= nBins_; i++){
   sprintf(s, "%-4d %-20.13e %-20.13e", i, xyM_[0][i], xyM_[1][i]);
   file << s << "\n";
  }  
  file.close();
}

// Print out the <x> and <y> momenta of bunch
void ICE::TImpedanceWF::showTKick(char* f)
{
  std::ofstream file;
  file.open(f);
  if(!file) {
    std::cerr << "ICE::TImpedanceWF::showTKick: Cannot open " << f << " for output \n";
    // _finalize_MPI();
    exit(1);
  }
 char s[120];
  int i;
  file << "nBins t_min t_max t_step [sec]:" << nBins_ 
       << "  " << t_min_ 
       << "  " << t_max_
       << "  " << t_step_ 
       <<  "\n"; 

  file << "i        xKick [rad]        yKick [rad]  \n";
  for (i = 0; i <= nBins_; i++){
   sprintf(s, "%-4d %-20.13e %-20.13e", i, tSumKick_ [0][i], 
                                           tSumKick_ [1][i]);
   file << s << "\n";
  }  
  file.close();
}

//Restore initial element's state
void ICE::TImpedanceWF::restore()
{
  tFFKick_[0]->zero();
  tFFKick_[1]->zero();
}



// Transverse Kick calculation
void ICE::TImpedanceWF::tKickCalc_( double t)
{

 // parameters needed : t_min_ ; t_max_ ; t_step_ ; z = xyM_[1..2][0..nBins_] 
 // wake_zero_[1..2] ; eta_fact_[1..2] ; eta_coff_[1..2]

 int i_xy, i;
 double z;
 for ( i_xy = 0; i_xy <= 1 ; i_xy++){
   if( xy_mask_[i_xy] == 1){

     eta_fact_[i_xy]->defShift(  *eta_coff_[i_xy], t_step_);

     tfKick_[i_xy]->copy(*tFFKick_[i_xy]);
     tfKick_[i_xy]->shift(  *eta_coff_[i_xy], (t - t_max_));

     tKick_[i_xy][nBins_]->zero();

      for( i = (nBins_-1) ; i >= 0 ; i--){
       tKick_[i_xy][i]->copy(*wake_zero_[i_xy]);       
       z = xyM_[i_xy][i+1];
       tKick_[i_xy][i]->multR(z);
       tKick_[i_xy][i]->sum(*tKick_[i_xy][i+1]);
       tKick_[i_xy][i]->mult(*eta_fact_[i_xy]);
       tKick_[i_xy][i+1]->sum(*tfKick_[i_xy]);
       tfKick_[i_xy]->mult(*eta_fact_[i_xy]);
      }

     tKick_[i_xy][0]->sum(*tfKick_[i_xy]);

     tfKick_[i_xy]->copy(*wake_zero_[i_xy]);     
     z = xyM_[i_xy][0];
     tfKick_[i_xy]->multR(z); 
     tfKick_[i_xy]->sum(*tKick_[i_xy][0]);
     tfKick_[i_xy]->shift(*eta_coff_[i_xy], t_min_);
     tFFKick_[i_xy]->copy(*tfKick_[i_xy]);

     for( i = 0 ; i <= nBins_ ; i++){
       tSumKick_ [i_xy][i] = coefficient_*tKick_[i_xy][i]->sumIm();      
     } 

   }
 }
}

//Propagate the bunch through the transvers impedance element 
void ICE::TImpedanceWF::propagate(PAC::Bunch& bunch , double t)
{
  // t - time after the previous bunch had passed through the element [sec]
  defineLongBin(bunch);

  if(bunchSize_ == 0 ) {return;}

  //if time is equal to zero we suppose that there is only one bunch
   if( t == 0.0) {
    tFFKick_[0]->zero();
    tFFKick_[1]->zero();
   }

  tKickCalc_(t);

  if( ((t+time_min_prev_ - t_max_ ) < 0.0) & (t != 0.0) ) {
    std::cerr << "ICE::TImpedanceWF::propagate : This and previous bunches are overlaped. Error. Stop.\n";
    // _finalize_MPI();
    exit(1);    
  } 

  int iT,ip;
  double  dp;
  double fract, fract1;
  for(ip = 0; ip < bunch.size(); ip++){
    if(!bunch[ip].isLost()) {
     iT = bins_[ip];
      fract = fractBins_[ip];
      fract1 = 1.0 - fract;

       // x-axis kick
       if( xy_mask_[0] == 1){
        dp = fract1*tSumKick_[0][iT]+fract*tSumKick_[0][iT+1];
        bunch[ip].getPosition().setPX(bunch[ip].getPosition().getPX() + dp);
       }

       // y-axis kick
       if( xy_mask_[1] == 1){
        dp = fract1*tSumKick_[1][iT]+fract*tSumKick_[1][iT+1];
        bunch[ip].getPosition().setPY(bunch[ip].getPosition().getPY() + dp);
       }          

    }
  }
  time_min_prev_ = t_min_;
}


//memorize the state of the transverse impedance element
void ICE::TImpedanceWF::memorizeState()
{
 int i_xy;
 for ( i_xy = 0; i_xy <= 1 ; i_xy++){
   if( xy_mask_[i_xy] == 1){
      tFFmemory_[i_xy]->copy(*tFFKick_[i_xy]);   
   }
 }
} 

//restore the memorized state of the transverse impedance element
void ICE::TImpedanceWF::restoreState()
{
 int i_xy;
 for ( i_xy = 0; i_xy <= 1 ; i_xy++){
   if( xy_mask_[i_xy] == 1){
      tFFKick_[i_xy]->copy(*tFFmemory_[i_xy]);   
   }
 }
}

// Get v0/c
double ICE::TImpedanceWF::getBeta_(const PAC::Bunch& bunch)
{
  double e = bunch.getBeamAttributes().getEnergy();
  double m = bunch.getBeamAttributes().getMass();

  return sqrt(e*e - m*m)/e;
}

// Synchronize longitudinal grid size for a parallel version (empty method)
// this method will be overridden in parallel version
void ICE::TImpedanceWF::_grid_synchronize(double& ct_min, double& ct_max)
{
}

// Sum longitudinal distribution for a parallel version (empty method)
// this method will be overridden in parallel version
void ICE::TImpedanceWF::_sum_distribution()
{
}





//===================debugging methods =========================
