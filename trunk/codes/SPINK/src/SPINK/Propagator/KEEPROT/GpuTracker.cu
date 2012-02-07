// Library       : SPINK
// File          : SPINK/Propagator/GpuTracker.cu
// Copyright     : see Copyright file
// Author        : V.Ranjbar
//  This the class which invokes the GPU kernel to perform spin and orbit
//  particle push using gpuPropagate kernel in the GpuPropagate C++ function
//  There is a lot of legacy functions using an older and much slower method where
//  all the pieces of the orbit and spin push are sperated out into individual kernel
//  calls. These are still in the code but unused. Perhaps later they can be removed.



#include "UAL/APF/PropagatorFactory.hh"
#include "PAC/Beam/Bunch.hh"
//#include "../../../../common/book.h"
#include "TEAPOT/Integrator/TrackerFactory.hh"
#include "TEAPOT/Integrator/DipoleData.hh"
#include "TEAPOT/Integrator/MagnetData.hh"
#include "SPINK/Propagator/GpuTracker_hh.cu"
#include "SPINK/Propagator/SpinTrackerWriter.hh" 

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <fstream>

//declaring global GPU variables 
__device__ vec6D pos_d[PARTICLES];
//__device__ vec6D tmp_d[PARTICLES];
__device__ Lat rhic_d[ELEMENTS];
//__shared__ Qlat MLT_d;
//Qlat MLT[ELEMENTS];
//#include <gpuKernels.cu>
#include <gpuProp11.cu>
SPINK::GpuTracker::GpuTracker()
{
  p_entryMlt = 0;
  p_exitMlt = 0;
  p_length = 0;
  p_bend = 0;
  p_mlt = 0;
  p_offset = 0;
  p_rotation = 0;
  // p_aperture = 0;
  p_complexity = 0;
  // p_solenoid = 0;
  // p_rf = 0;
  m_ir = 0.00;
}

bool SPINK::GpuTracker::coutdmp = 0;
int SPINK::GpuTracker::nturn = 0;
int SPINK::GpuTracker::Nelement = 0;
int SPINK::GpuTracker::threads = 1;

precision SPINK::GpuTracker::dthread = 0.0;
precision SPINK::GpuTracker::m_V = 0.00;
precision SPINK::GpuTracker::m_h = 0.00;
precision SPINK::GpuTracker::m_lag = 0.00;
precision SPINK::GpuTracker::circ = 0.00; 

precision SPINK::GpuTracker::snk1_mu = 0.00;
precision SPINK::GpuTracker::snk2_mu = 0.00;
precision SPINK::GpuTracker::snk1_phi = 0.00;
precision SPINK::GpuTracker::snk2_phi = 0.00;
precision SPINK::GpuTracker::snk1_theta = 0.00;
precision SPINK::GpuTracker::snk2_theta = 0.00;
precision SPINK::GpuTracker::stepsize = 0.1; //set default stepsize for quads

SPINK::GpuTracker::GpuTracker(const SPINK::GpuTracker& st)
{
  copy(st);
  m_data = st.m_data;
  m_mdata = st.m_mdata;
  m_ir = st.m_ir;
}

SPINK::GpuTracker::~GpuTracker()
{
}

UAL::PropagatorNode* SPINK::GpuTracker::clone()
{
  return new SPINK::GpuTracker(*this);
}


/** This Class reads in the lattice information into both the C++ classes
    and loads them into the GPU lattice variables.
**/

void SPINK::GpuTracker::setLatticeElements(const UAL::AcceleratorNode& sequence,
					   int is0, int is1,
					   const UAL::AttributeSet& attSet)
{
  
    SPINK::SpinPropagator::setLatticeElements(sequence, is0, is1, attSet);
 
    const PacLattice& lattice = (PacLattice&) sequence;
   
    setElementData(lattice[is0]);
   
    
    setConventionalTracker(sequence, is0, is1, attSet);
   
    m_name = lattice[is0].getName();
    /** loading up GPU lattice **/
    static int el=0;   
    double isMzero = 0;
    std::string m_type = lattice[is0].getType();

    // std::cout << "is0 = " << is0 << " el =" << el << " \n";
    /** setting rf flags indicating rfcavity rfcav=1 **/
    if(m_name == "rfac1"){  
    rhic[el].rfcav = 1;
    }else{rhic[el].rfcav = 0;}

   /** setting snake flag indicating snake1 or snake2 present **/
 rhic[el].snake = 0;
    if(m_name == "snake1"){
      rhic_d[el].snake = 1;
    }else if(m_name == "snake2") { rhic_d[el].snake = 2;
    }else if(m_name == "rot6") { rhic_d[el].snake = 3;
    }else if(m_name == "rot7"){rhic_d[el].snake = 4;
    }else if(m_name == "rot8"){rhic_d[el].snake = 5;
    }else if(m_name == "rot5"){rhic_d[el].snake = 6;}

    /** setting multipole values **/
    /** initializing to zero **/ 
   for(int k=0 ; k < 10 ; k++){
      rhic[el].entryMlt[k] = 0.0;
      rhic[el].exitMlt[k] = 0.0;
      rhic[el].mlt[k] =0.0;
      //   MLT[el].mlt[k] = 0.0;
    }
   

  if(p_entryMlt){
 
     int size_entry = p_entryMlt->size();
     double * data = p_entryMlt->data();
     for( int ii = 0; ii < size_entry; ii++)
       {  
	 /** pre-slicing up things to save time **/
	 if(!p_complexity){  
	   rhic[el].entryMlt[ii] = (precision) data[ii]/2.0; 
           isMzero += data[ii];   
          }
	 else{ 
	   //  int ns = 4*p_complexity->n();
	   rhic[el].entryMlt[ii] = (precision) data[ii]; // /(2.0*ns);
            isMzero += data[ii];  
            }
       } //end of for loop
  }else {   
    /** setting entryMlt[0]= 10000. indicates no entryMlt present **/ 
rhic[el].entryMlt[0] = 10000. ;}
   


 if(p_exitMlt){ 
  
     int size_exit = p_exitMlt->size();
     double * data = p_exitMlt->data();
     for( int ii = 0; ii < size_exit; ii++)
       {  
     	 /** pre-slicing up things to save time **/
       if(!p_complexity){ 
	   rhic[el].exitMlt[ii] = (precision) data[ii]/2.0;
 isMzero += data[ii];  
}
	 else {
	   // int ns = 4*p_complexity->n();
	   rhic[el].exitMlt[ii] = (precision) data[ii]; // /(2.0*ns);
            isMzero += data[ii];  
            }
	   
       }
 }else {
   /** setting exitMlt[0]=10000. indicates no exitMlit present **/
rhic[el].exitMlt[0] = 10000.;}




if(p_mlt){ 
      int sizemlt = p_mlt->size();
     double * data = p_mlt->data();
     for( int ii = 0; ii < sizemlt; ii++)
       {  
	 /** pre-slicing up things to save time **/
      if(!p_complexity){ 
	   rhic[el].mlt[ii] = (precision) data[ii]/2.0;
	   // MLT[el].mlt[ii] = (precision) data[ii]/2.0;
               isMzero += data[ii];  
}
	 else {
	   //  int ns = 4*p_complexity->n();
	   rhic[el].mlt[ii] = (precision) data[ii]; // /(2.0*ns);
	   // MLT[el].mlt[ii] = (precision) data[ii];
            isMzero += data[ii];  
            } 
       }
     rhic[el].order = p_mlt->order(); } else { 
  /** setting mlt[0] = 10000. indicates no mlt present **/
  rhic[el].mlt[0] = 10000.;   rhic[el].order = 0;
  // MLT[el].mlt[0] =  10000;
    }

/**setting m_l values length of element**/
 rhic[el].m_l = 0.;
  if(m_data.m_l) rhic[el].m_l = m_data.m_l;

/** setting bend transport values **/
   
/** initializing everyone to zero **/
   rhic[el].k1l = 0.;
   rhic[el].angle = 0.;
   rhic[el].btw01 = 0.;
   rhic[el].btw00 = 0.;
   rhic[el].atw01= 0.;
   rhic[el].atw00 = 0.;
   if(m_mdata.m_mlt){
     if(m_mdata.m_mlt->kl(1)) rhic[el].kl1 = m_mdata.m_mlt->kl(1);
     

    rhic[el].angle =  m_data.m_angle;
    rhic[el].btw01 = m_data.m_btw01;
    rhic[el].btw00 = m_data.m_btw00;
    rhic[el].atw01 = m_data.m_atw01;
    rhic[el].atw00 = m_data.m_atw00;}

for(int counter = 0; counter <= 1;counter++){
    rhic[el].cphpl[counter] = (precision) m_data.m_slices[counter].cphpl();
    rhic[el].sphpl[counter] = (precision) m_data.m_slices[counter].sphpl();
    rhic[el].tphpl[counter] = (precision) m_data.m_slices[counter].tphpl();
    rhic[el].scrx[counter] =  (precision) m_data.m_slices[counter].scrx();
    rhic[el].rlipl[counter] = (precision) m_data.m_slices[counter].rlipl();
    rhic[el].scrs[counter] =  (precision) m_data.m_slices[counter].scrs();
    rhic[el].spxt[counter] =  (precision) m_data.m_slices[counter].spxt();}
    int counter = -1;
     for(int i = 0; i < m_data.m_ir; i++){
   for(int is = 1; is < 5; is++){
     counter++;
    rhic[el].cphpl[counter] = (precision) m_data.m_slices[counter].cphpl();
    rhic[el].sphpl[counter] = (precision) m_data.m_slices[counter].sphpl();
    rhic[el].tphpl[counter] = (precision) m_data.m_slices[counter].tphpl();
    rhic[el].scrx[counter] =  (precision) m_data.m_slices[counter].scrx();
    rhic[el].rlipl[counter] = (precision) m_data.m_slices[counter].rlipl();
    rhic[el].scrs[counter] =  (precision) m_data.m_slices[counter].scrs();
    rhic[el].spxt[counter] =  (precision) m_data.m_slices[counter].spxt();
   
     }
}

/** setting complexity and ir values **/
 rhic[el].ns = 0;
 if(p_complexity){ rhic[el].ns= p_complexity->n();
   double ns = rhic[el].ns;
   if(p_mlt && ns > 2){
   double    ns_n = ns;
       precision leng = rhic[el].m_l;
       /*  if(stepsize >= leng/(4*ns)) {  
	 std::cout << "tripped this leng/ns = " << leng/(4*ns) << " with stepsize set to =" << stepsize << " \n";    
       fac = ns/4; ns = 4; leng = fac*leng; fac = 1;
       }*/
 while(leng/(ns_n*4) > stepsize) ns_n= ns_n+1.;
   double fac =  ns_n/ns; 
      leng = leng/fac;
      rhic[el].ns = (int) ns_n; rhic[el].m_l = leng;
      std::cout << "element number = " << el << " real step size = " << leng/(4*ns_n) <<  "set stepsize=" << stepsize << " ns_n = " << ns_n << "fac = " <<  fac <<  " ns = " << ns << " \n";  
 
   }}

rhic[el].m_ir = m_ir;

// if(m_data.m_ir > 0) std::cout << "m_ir not zero =" << m_data.m_ir << " \n";

/** setting for spin prop **/
// ns =1;
//  if(rhic[el].ns >0) ns = 4*rhic[el].ns;
 
   /** setting bend for spin prop **/
   rhic[el].bend = 0.0;
   if(p_bend) rhic[el].bend = p_bend->angle();
   /** setting for bend for spin prop **/ 
   rhic[el].length = 0.0;
  
   if(p_length){ rhic[el].length = p_length->l();
    }
 
/** setting offset values **/
 rhic[el].dx = 0.;
   rhic[el].dy = 0.;
   if(p_offset){
   rhic[el].dx = p_offset->dx();
   rhic[el].dy = p_offset->dy();
   }



     rhic[el].kl1 = 0.;
     rhic[el].k0l = 0.;
     rhic[el].kls0 = 0.;
     rhic[el].k2l = 0.;
     if(p_mlt){
       if(rhic[el].order == 0){
	 rhic[el].k0l = p_mlt->kl(0); // /ns;
	 rhic[el].kls0 = p_mlt->ktl(0); } // /ns;}

	 if(rhic[el].order > 0) rhic[el].k1l = p_mlt->kl(1); // /ns;
	 if(rhic[el].order > 1) rhic[el].k2l = p_mlt->kl(2); // /ns;
     }
      
     Nelement = el;
     isMzero += rhic[el].m_l + rhic[el].bend + rhic[el].snake + rhic[el].rfcav;
     if(isMzero != 0) el++;
     isMzero = 0.0;
}


void SPINK::GpuTracker::setLatticeElement(const PacLattElement& e)
{
  m_ir = e.getN();
  m_data.setLatticeElement(e);
  m_mdata.setLatticeElement(e);


}

void SPINK::GpuTracker::setConventionalTracker(const UAL::AcceleratorNode& sequence,
                                                int is0, int is1,
                                                const UAL::AttributeSet& attSet)
{
    const PacLattice& lattice = (PacLattice&) sequence;

    precision ns = 2;
    if(p_complexity) ns = 8*p_complexity->n();

    UAL::PropagatorNodePtr nodePtr =
      TEAPOT::TrackerFactory::createTracker(lattice[is0].getType());
     

    m_tracker = nodePtr;
    
    if(p_complexity) p_complexity->n() = 0;   // ir
    if(p_length)    *p_length /= ns;          // l
    if(p_bend)      *p_bend /= ns;            // angle, fint
     
    m_tracker->setLatticeElements(sequence, is0, is1, attSet);
   setLatticeElement(lattice[is0]);
   
  
    if(p_bend)      *p_bend *= ns;
    if(p_length)    *p_length *= ns;
    if(p_complexity) p_complexity->n() = ns/8;

}

/** Class for performing full spin orbit propagation on GPU **/
void SPINK::GpuTracker::GpuProp(PAC::Bunch& bunch)
{ 
static int firstcall = 0;
  int N = bunch.size();
  /** found this threadPerBlock size seemed to have best timing. Could experiment more to find a better number **/
  int threadsPerBlock = 128 ;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  // std::cout << "blocksPerGrid =" << blocksPerGrid << " \n";
  // std::cout << "threadsPerBlock =" << threadsPerBlock << "\n";
  //  threadsPerBlock = 1; blocksPerGrid = 1;
  
  /** if this is the first time we are calling this function then load up particles onto the GPU **/
 if(firstcall == 0){
   loadPart(bunch);
   }
    firstcall = 1;
    // I did some tests here trying to load the lattice into Constant Memory on 
    // the GPU since it wouldn't all fit into memory I tried breaking it appart
    // but this didn't seem to deliver any performance inprovement. I leave
    // it here commented out incase some one might wish to try it again.
 
    //   int npass = Nelement/20;
    //    int leftover = Nelement - npass*20; 
    // for(int turns= 1; turns <= nturn; turns++) {
    //for(int k=0;k < npass ;k++){
    // cudaMemcpyToSymbol(rhic_d,&rhic[k*20], sizeof(Lat)*20);
    gpuPropagate<<<blocksPerGrid, threadsPerBlock>>>(N,nturn,Nelement);
    // }
   

    //   if(leftover > 0){
    //    cudaMemcpyToSymbol(rhic_d,&rhic[npass*20], sizeof(Lat)*leftover);
    //    gpuPropagate<<<blocksPerGrid, threadsPerBlock>>>(N,leftover);
    //   }
    

    //  }
}




void SPINK::GpuTracker::copy(const SPINK::GpuTracker& st)
{
    m_name       = st.m_name;

    p_entryMlt   = st.p_entryMlt;
    p_exitMlt    = st.p_exitMlt;

    p_length     = st.p_length;
    p_bend       = st.p_bend;
    p_mlt        = st.p_mlt;
    p_offset     = st.p_offset;
    p_rotation   = st.p_rotation;
    // p_aperture = st.p_aperture;
    p_complexity = st.p_complexity;
    // p_solenoid = st.p_solenoid;
    // p_rf = st.p_rf;
}



/** Class for passing all vectors and parameters to GPU memory **/

void SPINK::GpuTracker::loadPart(PAC::Bunch& bunch)
{
  
    PAC::BeamAttributes& ba = bunch.getBeamAttributes();
    precision e0    =   (precision) ba.getEnergy();
    precision m0    =  (precision) ba.getMass();
    precision GG    =  (precision) ba.getG();
    precision q           =  (precision)  ba.getCharge();
    precision p0 = sqrt(e0*e0 - m0*m0);
    // precision gam = e0/m0;
    precision v0byc = p0/e0;
    precision Energy[PARTICLES],v0byc_c[PARTICLES],p0_c[PARTICLES];
    int N = bunch.size();
    precision dtr_h = atan(1.00)/45.00;
    //  cudaMemcpyToSymbol(p0_d,&p0,sizeof(precision));
    // cudaMemcpyToSymbol(Energy_d,&e0,sizeof(precision));
    cudaMemcpyToSymbol(GG_d,&GG,sizeof(precision));
    cudaMemcpyToSymbol(m0_d,&m0,sizeof(precision));
    cudaMemcpyToSymbol(q_d,&q,sizeof(precision));
    // cudaMemcpyToSymbol(stepsize_d,&stepsize,sizeof(precision));
    //   cudaMemcpyToSymbol(gam_d,&gam,sizeof(precision));
    //   cudaMemcpyToSymbol(v0byc_d,&v0byc,sizeof(precision));
    cudaMemcpyToSymbol(snk1_mu_d,&snk1_mu,sizeof(precision));
    cudaMemcpyToSymbol(snk1_theta_d,&snk1_theta,sizeof(precision));
    cudaMemcpyToSymbol(snk1_phi_d,&snk1_phi,sizeof(precision));
    cudaMemcpyToSymbol(snk2_mu_d,&snk2_mu,sizeof(precision));
    cudaMemcpyToSymbol(snk2_theta_d,&snk2_theta,sizeof(precision));
    cudaMemcpyToSymbol(snk2_phi_d,&snk2_phi,sizeof(precision));
    cudaMemcpyToSymbol(V_d,&m_V,sizeof(precision));
    cudaMemcpyToSymbol(h_d,&m_h,sizeof(precision));
    cudaMemcpyToSymbol(lag_d,&m_lag,sizeof(precision));
    cudaMemcpyToSymbol(circ_d,&circ,sizeof(precision));
    cudaMemcpyToSymbol(dtr,&dtr_h,sizeof(precision));
    cudaMemcpyToSymbol(rhic_d,rhic, sizeof(rhic));
    // cudaMemcpyToSymbol(MLT_d,MLT, sizeof(MLT));


  for(int ip = 0; ip <PARTICLES; ip++){
    pos[ip].sx = pos[ip].sy = pos[ip].sz = 0.00;
    pos[ip].x = pos[ip].y= pos[ip].ct = 0.00;
    pos[ip].px = pos[ip].py = pos[ip].de = 0.00;
    Energy[ip] = e0; p0_c[ip] = p0; v0byc_c[ip] = v0byc; 
}

  std::cout << "threads in loadPart =" << threads << "N = " << N << " \n";
  int Nb = N/threads;
  for(int it = 0 ; it < threads; it++) {
   for(int ip = 0; ip < Nb; ip++) {
      PAC::Position& part = bunch[ip+it*Nb].getPosition();
      pos[ip+it*Nb].sx = (precision)  bunch[ip+it*Nb].getSpin()->getSX();
      pos[ip+it*Nb].sy=  (precision) bunch[ip+it*Nb].getSpin()->getSY();
      pos[ip+it*Nb].sz =  (precision) bunch[ip+it*Nb].getSpin()->getSZ();

  
        
      pos[ip+it*Nb].x = (precision)  part.getX(); pos[ip+it*Nb].px=  (precision)  part.getPX(); pos[ip+it*Nb].y= (precision)  part.getY();
      pos[ip+it*Nb].py = (precision)  part.getPY(); pos[ip+it*Nb].ct = (precision)  part.getCT(); pos[ip+it*Nb].de=  (precision) part.getDE();
   
      Energy[ip+it*Nb] = e0 + dthread*it;
      std::cout << "ip =" << ip << "it =" << it << "Nb =" << Nb << "Energy =" << Energy[ip+it*Nb] << "\n";
      p0_c[ip+it*Nb] =  sqrt(Energy[ip+it*Nb]*Energy[ip+it*Nb] - m0*m0);
      v0byc_c[ip+it*Nb] =         p0_c[ip+it*Nb]/Energy[ip+it*Nb];
  
   }
  }

 // std::cout << "before sending to Gpu \n";
  cudaMemcpyToSymbol(pos_d,pos, sizeof(pos));
  // cudaMemcpyToSymbol(tmp_d,pos, sizeof(pos));
  cudaMemcpyToSymbol(Energy_d,Energy,sizeof(Energy));
  cudaMemcpyToSymbol(p0_d,p0_c,sizeof(p0_c));
  cudaMemcpyToSymbol(v0byc_d,v0byc_c,sizeof(v0byc_c));
 // std::cout << "after sending to Gpu \n";


}


/** Class to read back all particle vectors and parameters back to CPU 
    to be printed out **/

void SPINK::GpuTracker::readPart(PAC::Bunch& bunch,int printall)
{ int N = bunch.size();

 PAC::BeamAttributes& ba = bunch.getBeamAttributes();
  precision e0 = (precision)  ba.getEnergy(), m0 = (precision)  ba.getMass();
  precision gam ; //= e0/m0;
  // precision Energy[PARTICLES];
   precision GG    =  (precision) ba.getG();
   // precision Ggam  = gam*GG; 
   // precision SxAvg =0.00, SyAvg=0.00, SzAvg=0.00;
   // int count =0;
  cudaMemcpyFromSymbol(Energy,Energy_d, sizeof(Energy));
  // cudaMemcpyFromSymbol(&dS2,dS2_d,sizeof(precision));
  
    // cudaMemcpyFromSymbol(v0byc,v0byc_d,sizeof(v0byc));
  gam = Energy[0]/m0;
  e0 = Energy[0];
  printf(" gam = %e \n",gam);
  // dS2 = 0.0;
  // cudaMemcpyFromSymbol(&dS2_d,dS2,sizeof(precision));
  //    ba.setEnergy(e0);
  // precision Ggam  = gam*GG; 
//vec6D output[PARTICLES];
  cudaMemcpyFromSymbol(pos,pos_d, sizeof(pos));
  /**
  for(int ip = 0; ip < N; ip++) {
    if(printall==1){
        std::cout  << ip << " "<< gam << " " << Ggam << " " << pos[ip].x << " " << pos[ip].px << " " << pos[ip].y << " " << pos[ip].py << " " << pos[ip].ct << " " << pos[ip].de << " " << pos[ip].sx << " " << pos[ip].sy << " " << pos[ip].sz << " \n";}
     if(pos[ip].x*pos[ip].px*pos[ip].y*pos[ip].py*pos[ip].ct*pos[ip].de != pos[ip].x*pos[ip].px*pos[ip].y*pos[ip].py*pos[ip].ct*pos[ip].de ){ 
     }else {count++;
     SxAvg += pos[ip].sx; SyAvg += pos[ip].sy; SzAvg += pos[ip].sz;

     }
   }
  
   int ip = 0;
   // SxAvg = SxAvg/(N+1); SyAvg =SyAvg/(N+1); SzAvg = SzAvg/(N+1);
   //  std::cout << count << " " <<  gam << " " << Ggam << " " << SxAvg/count  << " " << SyAvg/count << " " << SzAvg/count << " " << pos[ip].x << " " << pos[ip].px << " " << pos[ip].y << " " << pos[ip].py << " " << pos[ip].ct << " " << pos[ip].de << "  \n";
   printf(" %i  %e  %e  %e  %e  %e  %e  %e  %e  %e  %e  %e \n",count,gam,Ggam,SxAvg/count,SyAvg/count,SzAvg/count,pos[ip].x,pos[ip].px,pos[ip].y,pos[ip].py,pos[ip].ct,pos[ip].de);

  **/

}






SPINK::GpuTrackerRegister::GpuTrackerRegister()
{
  UAL::PropagatorNodePtr dipolePtr(new SPINK::GpuTracker());
  UAL::PropagatorFactory::getInstance().add("SPINK::GpuTracker", dipolePtr);
}

static SPINK::GpuTrackerRegister theSpinkGpuTrackerRegister;



