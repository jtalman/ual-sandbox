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
__device__ vec6D tmp_d[PARTICLES];
__device__ Lat rhic_d[ELEMENTS];
#include <gpuKernels.cu>
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


void SPINK::GpuTracker::setLatticeElements(const UAL::AcceleratorNode& sequence,
					   int is0, int is1,
					   const UAL::AttributeSet& attSet)
{
  
    SPINK::SpinPropagator::setLatticeElements(sequence, is0, is1, attSet);
 
    const PacLattice& lattice = (PacLattice&) sequence;
   
    setElementData(lattice[is0]);
   
    
    setConventionalTracker(sequence, is0, is1, attSet);
   
    m_name = lattice[is0].getName();
    //  std::cout << "get Name =" << m_name << " \n"; 
    /** loading up GPU lattice **/

    /** setting rf flags **/
if(m_name == "rfac9bnc" || m_name == "rfac9mhz"){
      rhic[is0].rfcav = 1;
    }else{rhic[is0].rfcav = 0;}
/** setting snake flags **/
 rhic[is0].snake = 0;
    if(m_name == "snake1"){
      rhic[is0].snake = 1;
    }else if(m_name == "snake2") { rhic[is0].snake = 2;}

    /** setting multipole values **/
    for(int k=0 ; k < 10 ; k++){
      rhic[is0].entryMlt[k] = 0.0;
      rhic[is0].exitMlt[k] = 0.0;
      rhic[is0].mlt[k] =0.0;
    }
    // rhic[is0].ENTRY = 1;
    // rhic[is0].EXIT = 1;
    // rhic[is0].MULT = 1;

  if(p_entryMlt){
    //    rhic[is0].ENTRY = 1;
     int size_entry = p_entryMlt->size();
     double * data = p_entryMlt->data();
     for( int ii = 0; ii < size_entry; ii++)
       {  
	 /** pre-slicing up things to save time **/
	 if(!p_complexity){  
	   rhic[is0].entryMlt[ii] = (precision) data[ii]/2.0; }
	 else{ 
	   int ns = 4*p_complexity->n();
	   rhic[is0].entryMlt[ii] = (precision) data[ii]/(2.0*ns);
     
            }
       } //end of for loop
  }else {   // rhic[is0].ENTRY = 0; 
rhic[is0].entryMlt[0] = 10000. ;}
   


 if(p_exitMlt){ 
   //   rhic[is0].EXIT = 1;
     int size_exit = p_exitMlt->size();
     double * data = p_exitMlt->data();
     for( int ii = 0; ii < size_exit; ii++)
       {  
     	 /** pre-slicing up things to save time **/
       if(!p_complexity){ 
	   rhic[is0].exitMlt[ii] = (precision) data[ii]/2.0;}
	 else {
	     int ns = 4*p_complexity->n();
	   rhic[is0].exitMlt[ii] = (precision) data[ii]/(2.0*ns);
     
            }
	   
       }
 }else {// rhic[is0].EXIT = 0; 
rhic[is0].exitMlt[0] = 10000.;}




if(p_mlt){ 
  //  rhic[is0].MULT = 1;
      int sizemlt = p_mlt->size();
     double * data = p_mlt->data();
     for( int ii = 0; ii < sizemlt; ii++)
       {  
	 /** pre-slicing up things to save time **/
      if(!p_complexity){ 
	   rhic[is0].mlt[ii] = (precision) data[ii]/2.0;}
	 else {
	     int ns = 4*p_complexity->n();
	   rhic[is0].mlt[ii] = (precision) data[ii]/(2.0*ns);
     
            } 
       }
     rhic[is0].order = p_mlt->order(); } else { //rhic[is0].MULT = 0;
  rhic[is0].mlt[0] = 10000.;   rhic[is0].order = 0;}

/**setting m_l values **/
 rhic[is0].m_l = 0.;
  if(m_data.m_l) rhic[is0].m_l = m_data.m_l;

/** setting bend transport values **/
   
   rhic[is0].k1l = 0.;
   rhic[is0].angle = 0.;
   rhic[is0].btw01 = 0.;
   rhic[is0].btw00 = 0.;
   rhic[is0].atw01= 0.;
   rhic[is0].atw00 = 0.;
   if(m_mdata.m_mlt){
     if(m_mdata.m_mlt->kl(1)) rhic[is0].kl1 = m_mdata.m_mlt->kl(1);
     

    rhic[is0].angle =  m_data.m_angle;
    rhic[is0].btw01 = m_data.m_btw01;
    rhic[is0].btw00 = m_data.m_btw00;
    rhic[is0].atw01 = m_data.m_atw01;
    rhic[is0].atw00 = m_data.m_atw00;}

for(int counter = 0; counter <= 1;counter++){
    rhic[is0].cphpl[counter] = (precision) m_data.m_slices[counter].cphpl();
    rhic[is0].sphpl[counter] = (precision) m_data.m_slices[counter].sphpl();
    rhic[is0].tphpl[counter] = (precision) m_data.m_slices[counter].tphpl();
    rhic[is0].scrx[counter] =  (precision) m_data.m_slices[counter].scrx();
    rhic[is0].rlipl[counter] = (precision) m_data.m_slices[counter].rlipl();
    rhic[is0].scrs[counter] =  (precision) m_data.m_slices[counter].scrs();
    rhic[is0].spxt[counter] =  (precision) m_data.m_slices[counter].spxt();}
    int counter = -1;
     for(int i = 0; i < m_data.m_ir; i++){
   for(int is = 1; is < 5; is++){
     counter++;
    rhic[is0].cphpl[counter] = (precision) m_data.m_slices[counter].cphpl();
    rhic[is0].sphpl[counter] = (precision) m_data.m_slices[counter].sphpl();
    rhic[is0].tphpl[counter] = (precision) m_data.m_slices[counter].tphpl();
    rhic[is0].scrx[counter] =  (precision) m_data.m_slices[counter].scrx();
    rhic[is0].rlipl[counter] = (precision) m_data.m_slices[counter].rlipl();
    rhic[is0].scrs[counter] =  (precision) m_data.m_slices[counter].scrs();
    rhic[is0].spxt[counter] =  (precision) m_data.m_slices[counter].spxt();
   
     }
}

/** setting complexity and ir values **/
 rhic[is0].ns = 0;
   if(p_complexity) rhic[is0].ns= p_complexity->n();
rhic[is0].m_ir = m_ir;

/** setting for spin prop **/
     int ns =1;
     if(rhic[is0].ns >0) ns = 4*rhic[is0].ns;
 
   /** setting bend for spin prop **/
   rhic[is0].bend = 0.0;
   if(p_bend) rhic[is0].bend = p_bend->angle();
   /** setting for bend for spin prop **/ 
   rhic[is0].length = 0.0;
   //  rhic[is0].h = 0.0;
   if(p_length){ rhic[is0].length = p_length->l();
     //  rhic[is0].h  = rhic[is0].bend/rhic[is0].length;
    }
 
/** setting offset values **/
 rhic[is0].dx = 0.;
   rhic[is0].dy = 0.;
   if(p_offset){
   rhic[is0].dx = p_offset->dx();
   rhic[is0].dy = p_offset->dy();
   }


   //  int ns = 1;
   //if(p_complexity) ns = 4*p_complexity->n();

     rhic[is0].kl1 = 0.;
     rhic[is0].k0l = 0.;
     rhic[is0].kls0 = 0.;
     rhic[is0].k2l = 0.;
     if(p_mlt){
       if(rhic[is0].order == 0){
     rhic[is0].k0l = p_mlt->kl(0)/ns;
     rhic[is0].kls0 = p_mlt->ktl(0)/ns;}

       if(rhic[is0].order > 0) rhic[is0].k1l = p_mlt->kl(1)/ns;
       if(rhic[is0].order > 1) rhic[is0].k2l = p_mlt->kl(2)/ns;
     }
 
     Nelement = is0;
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
     

    // std::cout << "setting Convertional tracker Type = " <<  lattice[is0].getType() << "\n";
    m_tracker = nodePtr;
    
    if(p_complexity) p_complexity->n() = 0;   // ir
    if(p_length)    *p_length /= ns;          // l
    if(p_bend)      *p_bend /= ns;            // angle, fint
     
    m_tracker->setLatticeElements(sequence, is0, is1, attSet);
    // std::cout << "set latticeElement()  in GpuTRacker \n";
   setLatticeElement(lattice[is0]);
   
  
    if(p_bend)      *p_bend *= ns;
    if(p_length)    *p_length *= ns;
    if(p_complexity) p_complexity->n() = ns/8;

}


void SPINK::GpuTracker::DriftProp(PAC::Bunch& bunch)
{
PAC::BeamAttributes& ba = bunch.getBeamAttributes();
 int N = bunch.size();
 precision oldT = (precision)  ba.getElapsedTime();

 precision e0 = (precision) ba.getEnergy(), m0 = (precision) ba.getMass();
 precision p0 = sqrt(e0*e0 - m0*m0);

 precision v0byc = p0/e0;
 //std::cout << "in Gpu DriftTracker \n";
 precision m_l = (precision) m_data.m_l;
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  // std::cout << "threadsPerBlock =" << threadsPerBlock << "blocksPerGrid =" << blocksPerGrid << " \n";
  Copygpu<<<blocksPerGrid,threadsPerBlock>>>();
  // HANDLE_ERROR( cudaMemcpy(tmp_d,pos_d,sizeof(tmp_d),cudaMemcpyDeviceToDevice) );
  //  std::cout << "after Memcpy \n";   
  //readPart(bunch);
   makeVelocitygpu<<<blocksPerGrid,threadsPerBlock>>>(v0byc, N);
   //std::cout << "after make Velocity \n";   
   //readPart(bunch);
    makeRVgpu<<<blocksPerGrid,threadsPerBlock>>>( v0byc, e0,  p0, m0, N);
    //std::cout << "after makeRV \n";   
    //readPart(bunch);
   
  passDriftgpu<<<blocksPerGrid,threadsPerBlock>>>(m_l, v0byc, N,  6);
  //  std::cout << "after passDrift \n";   
  //readPart(bunch);
  //checkAperture(bunch);

  ba.setElapsedTime(oldT + m_l/v0byc/UAL::clight);

}



void SPINK::GpuTracker::BendProp(PAC::Bunch& bunch) {
  PAC::BeamAttributes& ba = bunch.getBeamAttributes();
  PAC::Position& pos = bunch[0].getPosition();
  int N = bunch.size();
  precision oldT = (precision) ba.getElapsedTime();
  precision e0 = (precision) ba.getEnergy(), m0 = (precision) ba.getMass();
  precision p0 =  sqrt(e0*e0 - m0*m0);
  precision v0byc = p0/e0;
  precision dx = 0.0, dy = 0.0;
  precision gam = e0/m0;
  precision entry[10], vexit[10], mlt[10];
  // precision * entry, *vexit, *mlt;
  int size_entry,size_mlt,size_exit;
  precision t0 = oldT;
  //entry = (precision*)malloc(10*sizeof(precision));
  // vexit =  (precision*)malloc(10*sizeof(precision));
  //  mlt = (precision*)malloc(10*sizeof(precision));
  // std::cout << "top of BendProp \n";
  for( int ii = 0; ii < 10; ii++) {
      entry[ii] = 0.00; vexit[ii]=0.00;mlt[ii] =0.00;}

 int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
   Copygpu<<<blocksPerGrid,threadsPerBlock>>>();
  //HANDLE_ERROR( cudaMemcpyToSymbol(tmp_d,pos_d,sizeof(pos_d),cudaMemcpyDeviceToDevice) );
	if(m_mdata.m_entryMlt){
     size_entry = m_mdata.m_entryMlt->size();
     double * data = m_mdata.m_entryMlt->data();
     for( int ii = 0; ii < size_entry; ii++)
       {    entry[ii] = (precision) data[ii];
     //	 	 std::cout << "ii = " << ii << "entry =" << entry[ii] << "\n";
       }
     // free(data);
	
	//std::cout << "before applyMltkickgpu \n";
	//  readPart(bunch);  
     applyMltKickgpu<<<blocksPerGrid,threadsPerBlock>>>(entry[0],entry[1],entry[2],entry[3],entry[4],entry[5],0,0,1,N,size_entry);
     //std::cout << "after applyMltkickgpu \n";
     //     readPart(bunch);  
	}
      makeVelocitygpu<<<blocksPerGrid,threadsPerBlock>>>(v0byc, N);
      //std::cout << "after makeVel \n";
      //    readPart(bunch);  
      makeRVgpu<<<blocksPerGrid,threadsPerBlock>>>( v0byc, e0,  p0, m0, N);
      //std::cout << "after makeRVgpu \n";
      //   readPart(bunch);  	

       // Begin PassBend Section 
      precision kl1 = 0.00;

   if(m_mdata.m_mlt){
    if(m_mdata.m_mlt->kl(1)){
      precision kl1 = (precision) m_mdata.m_mlt->kl(1);}
}
     precision angle =  (precision) m_data.m_angle;
     precision btw01 =  (precision) m_data.m_btw01;
     precision btw00 =  (precision) m_data.m_btw00;
     precision atw01 =  (precision) m_data.m_atw01;
     precision atw00 =  (precision) m_data.m_atw00;
     precision m_l =     (precision) m_data.m_l;

if(m_mdata.m_offset){
       dx =  m_mdata.m_offset->dx();
       dy =  m_mdata.m_offset->dy();}
      
if(!m_data.m_ir){
         
	  precision cphpl =  (precision) m_data.m_slices[0].cphpl();
        
	  precision sphpl =  (precision) m_data.m_slices[0].sphpl();
         
	  precision tphpl =  (precision) m_data.m_slices[0].tphpl();
         
	  precision scrx =   (precision) m_data.m_slices[0].scrx();
         
	  precision rlipl =  (precision) m_data.m_slices[0].rlipl();
         
	  precision scrs =   (precision) m_data.m_slices[0].scrs();
         
	  precision spxt =   (precision) m_data.m_slices[0].spxt();
    
	   // std::cout << "before passBendSlicegpu \n";
	   //  std::cout << "inputs =" << cphpl << " " << sphpl << " " << tphpl << " " << scrx << " " << rlipl << " " << scrs << " " << spxt <<  " " << v0byc <<"\n";
	   //  readPart(bunch);     

	   passBendSlicegpu<<<blocksPerGrid,threadsPerBlock>>>(cphpl, sphpl, tphpl, scrx, scrs, spxt, rlipl, v0byc,N );
	   //   std::cout << "after passBendSlicegpu \n";
	   // readPart(bunch);     

      if(m_mdata.m_mlt){
  size_mlt   = (int) m_mdata.m_mlt->size();
       double * data2 = m_mdata.m_mlt->data();
     for( int ii = 0; ii < size_mlt; ii++)
       {    mlt[ii] = (precision) data2[ii];
	 // std::cout << "ii = " << ii << "in bend mlt =" << mlt[ii] << "\n";

 } 
     // free(data2);



     applyMltKickgpu<<<blocksPerGrid,threadsPerBlock>>>(mlt[0],mlt[1],mlt[2],mlt[3],mlt[4],mlt[5],dx,dy,1,N,size_mlt);}
      //  std::cout << "after applyMltKickgpu \n";
      //  readPart(bunch);
        
      //   std::cout << "before applythinbend \n";
      applyThinBendKickgpu<<<blocksPerGrid,threadsPerBlock>>>(v0byc, m_l, kl1, angle, btw01, btw00, atw01, atw00, dx, dy, 1.00,  N);
      //  std::cout << "after applyThinBendKickgpu \n";
      // readPart(bunch);
      makeVelocitygpu<<<blocksPerGrid,threadsPerBlock>>>(v0byc, N);
      // std::cout << "after makeVel \n";
      //readPart(bunch);

         cphpl =  (precision) m_data.m_slices[1].cphpl();
       
	 sphpl =  (precision) m_data.m_slices[1].sphpl();
       
	 tphpl =  (precision) m_data.m_slices[1].tphpl();
       
 	 scrx =   (precision) m_data.m_slices[1].scrx();
     
         rlipl =  (precision) m_data.m_slices[1].rlipl();
      
         scrs =  (precision) m_data.m_slices[1].scrs();
      
         spxt =  (precision) m_data.m_slices[1].spxt();
     
      passBendSlicegpu<<<blocksPerGrid,threadsPerBlock>>>(cphpl, sphpl, tphpl, scrx, scrs, spxt, rlipl, v0byc, N);
      // std::cout << "after 2nd passBendSlicegpu \n";
      //	     readPart(bunch);   

 } else {

 // Complex Element

 precision rIr = 1./m_data.m_ir;
 precision rkicks = 0.25*rIr;

 int counter = -1;
 for(int i = 0; i < m_data.m_ir; i++){
   for(int is = 1; is < 5; is++){
     counter++;
    
     precision cphpl = (precision) m_data.m_slices[counter].cphpl();
     precision sphpl = (precision) m_data.m_slices[counter].sphpl();
     precision tphpl = (precision) m_data.m_slices[counter].tphpl();
     precision scrx =  (precision) m_data.m_slices[counter].scrx();
     precision rlipl = (precision) m_data.m_slices[counter].rlipl();
     precision scrs =  (precision) m_data.m_slices[counter].scrs();
     precision spxt =  (precision) m_data.m_slices[counter].spxt();
     // std::cout << "before complex passBendSlice \n";
     // std::cout << "inputs =" << cphpl << " " << sphpl << " " << tphpl << " " << scrx << " " << rlipl << " " << scrs << " " << spxt  << " \n";
     //  readPart(bunch);  
     passBendSlicegpu<<<blocksPerGrid,threadsPerBlock>>>(cphpl, sphpl, tphpl, scrx, scrs, spxt, rlipl, v0byc, N);
     // std::cout << "after complex passBendSlicegpu no =" << i << " " << is << "\n";
     // readPart(bunch);   
      if(m_mdata.m_mlt){
    size_mlt   = (int) m_mdata.m_mlt->size();
      double * data2 = m_mdata.m_mlt->data();
     for( int ii = 0; ii < size_mlt; ii++)
       {    mlt[ii] = (precision) data2[ii];
	 // std::cout << "ii = " << ii << "in complex bend mlt =" << mlt[ii] << "\n";
     }
     // free(data2);
        applyMltKickgpu<<<blocksPerGrid,threadsPerBlock>>>(mlt[0],mlt[1],mlt[2],mlt[3],mlt[4],mlt[5],dx,dy,rkicks,N,size_mlt);
	//	std::cout << "after applyMltKick in complex \n";
	//	readPart(bunch);

      }
      // std::cout << "before 2nd applyThinBend \n";
        applyThinBendKickgpu<<<blocksPerGrid,threadsPerBlock>>>(v0byc, m_l, kl1, angle, btw01, btw00, atw01, atw00, dx, dy, rkicks,  N);
	//	std::cout << "after in complex applyThinBendKickgpu \n";
	//  readPart(bunch);  	


      makeVelocitygpu<<<blocksPerGrid,threadsPerBlock>>>(v0byc, N);

      //std::cout << "after makeVel in complex \n";
      //    readPart(bunch);  

   }
   counter++;
   precision cphpl =   (precision) m_data.m_slices[counter].cphpl();
   precision sphpl =   (precision) m_data.m_slices[counter].sphpl();
   precision tphpl =   (precision) m_data.m_slices[counter].tphpl();
   precision scrx =   (precision) m_data.m_slices[counter].scrx();
   int rlipl =   (precision) m_data.m_slices[counter].rlipl();
   precision scrs =   (precision) m_data.m_slices[counter].scrs();
   precision spxt =   (precision) m_data.m_slices[counter].spxt();
 
   //std::cout << "inputs =" << cphpl << " " << sphpl << " " << tphpl << " " << scrx << " " << rlipl << " " << scrs << " " << spxt  << " \n";

   passBendSlicegpu<<<blocksPerGrid,threadsPerBlock>>>(cphpl, sphpl, tphpl, scrx, scrs, spxt, rlipl, v0byc, N);
   //std::cout << "after passBendSlicegpu \n";
   // readPart(bunch);   

   makeVelocitygpu<<<blocksPerGrid,threadsPerBlock>>>(v0byc, N);
   //std::cout << "after makeVel \n";
   //     readPart(bunch);  


 }
 }
//End of Pass Bend Section
 if(m_mdata.m_exitMlt){
   size_exit  = (int) m_mdata.m_exitMlt->size();
   double * data3 = m_mdata.m_exitMlt->data();
     for( int ii = 0; ii < size_exit; ii++)
       {   vexit[ii] = (precision)  data3[ii]; 
	 // std::cout << "ii = " << ii << "in bend mlt exit =" << vexit[ii] << "\n";
   }

     // free(data3);
     //std::cout << "before applyMltKick \n";
     // readPart(bunch);  
    applyMltKickgpu<<<blocksPerGrid,threadsPerBlock>>>(vexit[0],vexit[1],vexit[2],vexit[3],vexit[4],vexit[5],0.00,0.00,1.00,N,size_exit);
    //std::cout << "after applyMltKick \n";
    //    readPart(bunch);  

 }

 ba.setElapsedTime((double) oldT + m_data.m_l/v0byc/UAL::clight);

 //free(mlt); free(vexit); free(entry);     
 return;

}

void SPINK::GpuTracker::MultProp(PAC::Bunch& bunch){
PAC::BeamAttributes& ba = bunch.getBeamAttributes();
  PAC::Position& pos = bunch[0].getPosition();
  int N = bunch.size();
  precision oldT =  (precision) ba.getElapsedTime();
  precision e0 = (precision)  ba.getEnergy(), m0 = (precision)  ba.getMass();
  precision p0 =    sqrt(e0*e0 - m0*m0);
  precision v0byc = p0/e0;
  precision dx = 0.00, dy = 0.00;
  precision length = 0;
  precision gam = e0/m0;
  precision entry[10], vexit[10], mlt[10];
  int size_entry,size_mlt,size_exit;
 
  // entry = (precision*)malloc(10*sizeof(precision));
  // vexit =  (precision*)malloc(10*sizeof(precision));
  //  mlt = (precision*)malloc(10*sizeof(precision));
  for( int ii = 0; ii < 10; ii++) {
    entry[ii] = 0.00; vexit[ii]=0.00;mlt[ii] =0.00;}

   if(m_mdata.m_offset){
       dx =  m_mdata.m_offset->dx();
       dy =  m_mdata.m_offset->dy();}


 int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
   Copygpu<<<blocksPerGrid,threadsPerBlock>>>();
   //HANDLE_ERROR( cudaMemcpyToSymbol(tmp_d,pos_d,sizeof(pos_d),cudaMemcpyDeviceToDevice) );
 precision m_l =   (precision) m_data.m_l;
 //std::cout << "start of multipole \n";
	
    if(m_mdata.m_entryMlt){
     size_entry = m_mdata.m_entryMlt->size();
     double * data = m_mdata.m_entryMlt->data();
     for( int ii = 0; ii < size_entry; ii++)
       {    entry[ii] =  (precision) data[ii];
	 //std::cout << "ii = " << ii << "in mutlipole part entry =" << entry[ii] << "\n";
     }
     // free(data);

	
     // std::cout << "before entry applyMltKickgpu \n";
     //readPart(bunch);
     applyMltKickgpu<<<blocksPerGrid,threadsPerBlock>>>(entry[0],entry[1],entry[2],entry[3],entry[4],entry[5],0.00,0.00,1.00,N,size_entry);}
    // std::cout << "after entry applyMltKickgpu \n";
    // readPart(bunch);

     makeVelocitygpu<<<blocksPerGrid,threadsPerBlock>>>(v0byc, N);
     //std::cout << "after makeVel \n";
     // readPart(bunch);
     makeRVgpu<<<blocksPerGrid,threadsPerBlock>>>( v0byc, e0,  p0, m0, N);
     //std::cout << "after makeRVgpu \n";
     //	     readPart(bunch);
     ///if simple 
     if(!m_ir){

     passDriftgpu<<<blocksPerGrid,threadsPerBlock>>>(m_l/2., v0byc, N,  6);
     //std::cout << "after passDriftgpu \n";
     //	     readPart(bunch);   
   
   if(m_mdata.m_mlt){
    size_mlt   = (int) m_mdata.m_mlt->size();
       double * data2 = m_mdata.m_mlt->data();
     for( int ii = 0; ii < size_mlt; ii++)
       {    mlt[ii] =   (precision) data2[ii];
	 // std::cout << "ii = " << ii << "in multipole part mlt =" << mlt[ii] << "\n";

       }
     // free(data2);   


     applyMltKickgpu<<<blocksPerGrid,threadsPerBlock>>>(mlt[0],mlt[1],mlt[2],mlt[3],mlt[4],mlt[5],dx,dy,1.00,N,size_mlt);}
   //std::cout << "after applyMltKick \n";
     //  readPart(bunch);
     makeVelocitygpu<<<blocksPerGrid,threadsPerBlock>>>(v0byc,N);
     // std::cout << "after makeVel \n";
     //  readPart(bunch); 
    passDriftgpu<<<blocksPerGrid,threadsPerBlock>>>(m_l/2.00, v0byc, N,  6);
    // std::cout << "after passDrift \n";
    //	     readPart(bunch);
 if(m_mdata.m_exitMlt){
   size_exit  = (int) m_mdata.m_exitMlt->size();
     double * data3 = m_mdata.m_exitMlt->data();
     for( int ii = 0; ii < size_exit; ii++)
       {    vexit[ii] = (precision) data3[ii];
	 //std::cout << "ii = " << ii << "in multipole vexit =" << vexit[ii] << "\n";
 }

     // free(data3);


     applyMltKickgpu<<<blocksPerGrid,threadsPerBlock>>>(vexit[0],vexit[1],vexit[2],vexit[3],vexit[4],vexit[5],0.00,0.00,1.00,N,size_exit);}
 //   std::cout << "after applyMltKick \n";
 // readPart(bunch);


     }       

    precision rIr = 1.00/m_ir;
    precision rkicks = 0.25*rIr;
    precision s_steps[] = {0.10, 4.00/15, 4.00/15, 4.00/15, 0.10};
    int counter = 0;
    for(int i = 0; i < m_ir; i++){
      for(int is = 0; is < 4; is++){
	counter++;
	//std::cout << "before passDriftgpu " << " i = " << i << " is = " << is << "\n";
	// readPart(bunch);
  passDriftgpu<<<blocksPerGrid,threadsPerBlock>>>(m_l*s_steps[is]*rIr, v0byc, N,  6);
  //std::cout << "after passDrift \n";
  //  readPart(bunch);

 if(m_mdata.m_mlt){
    size_mlt   = (int) m_mdata.m_mlt->size();
      double * data2 = m_mdata.m_mlt->data();
     for( int ii = 0; ii < size_mlt; ii++)
       {    mlt[ii] = (precision) data2[ii]; 
	 //	 std::cout << "ii = " << ii << "in multipole complex mlt =" << mlt[ii] << "\n";

}
     // free(data2); 

  

 //std::cout << "before applyMltKick \n";
 //  readPart(bunch);
     applyMltKickgpu<<<blocksPerGrid,threadsPerBlock>>>(mlt[0],mlt[1],mlt[2],mlt[3],mlt[4],mlt[5],dx,dy,rkicks,N,size_mlt);  }
  //std::cout << "after applyMltKick \n";
  //readPart(bunch);
  makeVelocitygpu<<<blocksPerGrid,threadsPerBlock>>>(v0byc,N);
  // std::cout << "after makeVel \n";
  //   readPart(bunch);
      }
      counter++;
      // std::cout << "before passDriftgpu \n";
      //	     readPart(bunch);
    passDriftgpu<<<blocksPerGrid,threadsPerBlock>>>(m_l*s_steps[4]*rIr, v0byc, N,  6);
    //std::cout << "after passDriftgpu \n";
    //    readPart(bunch);  

  }
if(m_mdata.m_exitMlt){
   size_exit  = (int) m_mdata.m_exitMlt->size();
     double * data3 = m_mdata.m_exitMlt->data();
     for( int ii = 0; ii < size_exit; ii++)
       {    vexit[ii] = (precision) data3[ii];
	 //std::cout << "ii = " << ii << "in multipole exit=" << vexit[ii] << "\n";

 }

     // free(data3);


// std::cout << "before applyMltKick \n";
//  readPart(bunch);
   applyMltKickgpu<<<blocksPerGrid,threadsPerBlock>>>(vexit[0],vexit[1],vexit[2],vexit[3],vexit[4],vexit[5],0.00,0.00,1.00,N,size_exit);
}
   // std::cout << "after applyMltKick \n";
   //     readPart(bunch);


 ba.setElapsedTime((double) oldT + length/v0byc/UAL::clight);

 //free(entry); free(mlt); free(vexit);
 return;


}



void SPINK::GpuTracker::RFProp(PAC::Bunch& bunch)
{
 PAC::BeamAttributes& ba = bunch.getBeamAttributes();
 // std::cout << " in SPINK GPU rf cavity tracker \n";
 int N = bunch.size();
 precision q           =  (precision)  ba.getCharge();
 precision m0          =  (precision) ba.getMass();
 precision e0_old      =  (precision) ba.getEnergy();
 precision p0_old      =   sqrt(e0_old*e0_old - m0*m0);
 precision v0byc_old   =   p0_old/e0_old;
 // precision revfreq_old =   ba.getRevfreq();
 precision t_old       =   (precision) ba.getElapsedTime();
  // RF attributes
  
  precision V   = m_V;
  precision lag = m_lag;
  precision h   = m_h;

 


  // Update the synchronous particle (beam attributes)

  precision de0       = q*V*sin(2*UAL::pi*lag);
  precision e0_new    = e0_old + de0;
  precision p0_new    = sqrt(e0_new*e0_new - m0*m0);
  precision v0byc_new = p0_new/e0_new;




  precision revfreq_old = v0byc_old*UAL::clight/circ ;
  ba.setEnergy(e0_new);
  ba.setRevfreq(revfreq_old*v0byc_new/v0byc_old);
   // Tracking
 precision m_l =   (precision) m_data.m_l;
 
 /**
     cout << "\nq=" << q << ", m0=" << m0 << ", e0_old=" << e0_old << endl;
  cout << "p0_old=" << p0_old << ", v0byc_old=" << v0byc_old << endl;
  cout << "revfreq_old=" << revfreq_old << ", t_old=" << t_old << endl;
  cout << "circ=" << circ << ", revfrteq_old=" << revfreq_old << endl ;

  cout << "V=" << V << ", lag=" <<  lag << " m_l =" << m_l << endl;
 **/
 // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    gpuRFTracker<<<blocksPerGrid, threadsPerBlock>>>(N,lag,p0_old,e0_old,m0,m_l,v0byc_old,p0_new,v0byc_new,h,q,V,revfreq_old,e0_new);

 
  ba.setElapsedTime(t_old + (m_l/v0byc_old + m_l/v0byc_new)/2./UAL::clight);

  return;
}


void SPINK::GpuTracker::GpuProp(PAC::Bunch& bunch)
{ 
static int firstcall = 0;
  int N = bunch.size();
  int threadsPerBlock = 100;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  // std::cout << "blocksPerGrid =" << blocksPerGrid << " \n";
  // std::cout << "threadsPerBlock =" << threadsPerBlock << "\n";
  //  threadsPerBlock = 1; blocksPerGrid = 1;
  
 if(firstcall == 0){
   loadPart(bunch);
   }
    firstcall = 1;
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



void SPINK::GpuTracker::propagate(UAL::Probe& b)
{
  PAC::Bunch& bunch = static_cast<PAC::Bunch&>(b);
  PAC::BeamAttributes& ba = bunch.getBeamAttributes();
  PAC::Position& pos = bunch[0].getPosition();
  int N = bunch.size();
  precision oldT =  (precision) ba.getElapsedTime();
  precision e0 =  (precision) ba.getEnergy(), m0 = (precision)  ba.getMass();
  precision p0 =    sqrt(e0*e0 - m0*m0);
  static int firstcall = 0;
  precision length = 0.00;
  precision gam = e0/m0;
   precision light = (precision) UAL::clight;
  precision v = p0/gam/m0*light;
  precision t0 = oldT;
 
   if(firstcall == 0){
   loadPart(bunch);
   }
    firstcall = 1;

  

  //std::cout << "element " << m_name << " ";
  //std::cout <<"p_mlt =" << p_mlt << "p_bend =" << p_bend << "\n";
  // std::cout << " 1st call \n";
  // readPart(bunch);
  double ang = 0.00;
  if(p_bend) ang = fabs(p_bend->angle());

  if(p_length){     length = p_length->l();
    //  std::cout << "length =" << length << "\n";
  }
  if(p_complexity){
    //  std::cout << "complex =" << p_complexity->n() << " \n";
  }
  if(!p_complexity){
    //   std::cout << "simple elements \n";

     length /= 2.00;
   
    if(p_mlt) *p_mlt /= 2.00;
    

// if(m_name == "rfac1") {
    if(m_name == "rfac9bnc" || m_name == "rfac9mhz" ) {
 //std::cout << "start of RFProp() 1 complex \n";
	RFProp(bunch);
	//	std::cout << "after RFProp() \n";
	// readPart(bunch);
	return;
 }


    if(ang > 0.00){
	// pick GPU Dipole propogator //
  	//std::cout << "start of Dipole Bend \n";
        BendProp(bunch);
	//	std::cout << "after BendProp \n";
	//	 readPart(bunch);
 }else if(p_mlt) {
	// pick GPU Multipole propagator //
      //std::cout << "start of Multipole prop \n";
	  MultProp(bunch);
	  //	  std::cout << "after Multipole Propo \n";
	  //  readPart(bunch); 
 } else if( length > 0){
      //std::cout << " called Drift prop \n";
      DriftProp(bunch);
      //std::cout << " after Drift prop \n";
      //readPart(bunch);
  }


      // m_tracker->propagate(bunch);

   
    

     if(p_mlt) *p_mlt *= 2.00;     
     t0 += length/v;
    ba.setElapsedTime(t0);
 
  
 if(m_name == "snake1" || m_name == "snake2"){
   //std::cout << "calling snake prop. \n";
      SnakeProp(bunch);} else{
      propagateSpin(bunch);}


    if(p_mlt) *p_mlt /= 2.00; 
    if(ang > 0.00){
	// pick GPU Dipole propogator //
      //	std::cout << "start of 2nd Dipole Bend \n";
        BendProp(bunch);
	//std::cout << "after BendProp \n";
        //readPart(bunch);
 }else if(p_mlt) {
	// pick GPU Multipole propagator //
      // std::cout << "start of 2nd Multipole prop \n";
	  MultProp(bunch);
	  //	  std::cout << "after Multipole Propo \n";
          //readPart(bunch); 
 } else if( length > 0){
      //std::cout << " called 2nd Drift prop \n";
      DriftProp(bunch);
      // std::cout << " after Drift prop \n";
      //readPart(bunch);
}
   
    //m_tracker->propagate(bunch);
    if(p_mlt) *p_mlt *= 2.00;  

 t0 += length/v;
    ba.setElapsedTime(t0);

    return;
  }

  //std::cout << " in complex element  ang = " << ang << " \n";
   int ns = 4*p_complexity->n();

   length /= 2*ns;

  for(int i=0; i < ns; i++) {

   if(p_mlt) *p_mlt /= (2*ns);          // kl, kt
   if(ang > 0.00){
	// pick GPU Dipole propogator //
     //	std::cout << "start of Dipole Bend \n";
        BendProp(bunch);
	//	std::cout << "after BendProp \n";
        //readPart(bunch);
 }else if(p_mlt) {
	// pick GPU Multipole propagator //
     //std::cout << "start of Multipole prop \n";
	  MultProp(bunch);
	  // std::cout << "after Multipole Propo \n";
	  // readPart(bunch); 
 }  else if( length > 0){
     //std::cout << " called Drift prop \n";
      DriftProp(bunch);
      //  std::cout << " after Drift prop \n";
      //   readPart(bunch);
   }
    //   m_tracker->propagate(bunch);
   if(p_mlt) *p_mlt *= (2*ns);          // kl, kt
 
    t0 += length/v;
    ba.setElapsedTime(t0);
    if(m_name == "snake1" || m_name == "snake2"){
      // std::cout << "calling snake prop. \n";
      SnakeProp(bunch);} else{
      propagateSpin(bunch);}

    if(p_mlt) *p_mlt /= (2*ns);          // kl, kt

    //std::cout << "2nd Complex Prop \n";
    if(ang > 0.00){
	// pick GPU Dipole propogator //
      //	std::cout << "start of Dipole Bend \n";
        BendProp(bunch);
	//	std::cout << "after BendProp \n";
	// readPart(bunch);
 }else if(p_mlt) {
	// pick GPU Multipole propagator //
     // std::cout << "start of Multipole prop \n";
	  MultProp(bunch);
	  //	  std::cout << "after Multipole Propo \n";
          //readPart(bunch); 
 } else if( length > 0){
     //std::cout << " called Drift prop \n";
      DriftProp(bunch);
      // std::cout << " after Drift prop \n";
      // readPart(bunch);
   }
   //  m_tracker->propagate(bunch);
   if(p_mlt) *p_mlt *= (2*ns);          // kl, kt
 
    t0 += length/v;
    ba.setElapsedTime(t0);

  }
  //  readPart(bunch);

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

void SPINK::GpuTracker::propagateSpin(UAL::Probe& b)
{
PAC::Bunch& bunch = static_cast<PAC::Bunch&>(b);
    
    PAC::BeamAttributes& ba = bunch.getBeamAttributes();
    int N = bunch.size();
    precision e0    =   (precision) ba.getEnergy();
    precision m0    =  (precision) ba.getMass();
    precision GG    =  (precision) ba.getG();
    precision p0    = sqrt(e0*e0 - m0*m0);
    precision v0byc = p0/e0;
 int ns = 1;
  if(p_complexity) ns = 4*p_complexity->n();

  precision length = 0.00;
  if(p_length) length = p_length->l()/ns;

  precision ang = 0.00, h = 0.00;
  if(p_bend)  {
      ang    = p_bend->angle()/ns;
// VR added to avoid div by zero Dec22 2010
     if(length > 1e-20)
      h      = ang/length;
  }

precision k1l = 0.00, k2l = 0.00;
precision k0l = 0.00, kls0 = 0.00; // VR added to handle hkicker and vkicker spin effects Dec22 2010
  if(p_mlt){
    if(p_mlt->order() == 0){
      k0l = p_mlt->kl(0)/ns; kls0 = p_mlt->ktl(0)/ns ;} // VR added to handle hkicker and vkicker spin effects Dec22 2010
    if(p_mlt->order() > 0) k1l   =   p_mlt->kl(1)/ns;
    if(p_mlt->order() > 1) k2l   =   p_mlt->kl(2)/ns;
  }

 

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

 if(length != 0 || k1l != 0 || k2l != 0) {
   gpupropogateSpin<<<blocksPerGrid, threadsPerBlock>>>(N, ang, p0, e0, m0, h, length, k1l, k2l, k0l,kls0,v0byc, GG);
    // std::cout << "coming back from Spin GPU push \n";
    // std::cout << "Spin after push \n";

 } //else { std::cout << "no precessing fields \n";}
    

}

void SPINK::GpuTracker::SnakeProp(PAC::Bunch& bunch)
{
 PAC::BeamAttributes& ba = bunch.getBeamAttributes();
      int N;
  N = bunch.size();
   precision dtr = atan(1.00)/45.00;
  

  precision A[3] ;
  precision s_mat[9] ;

 
  if( m_name == "snake1") {
    //   std::cout << "doing snake1 prop \n";
    precision cs = 1.00 -cos(snk1_mu*dtr) ; precision sn =  sin(snk1_mu*dtr) ;

    A[0] = cos(snk1_theta*dtr) * sin(snk1_phi*dtr) ; // a(1) in MaD-SPINk
    A[1] = sin(snk1_theta*dtr) ;                // a(2) in MAD-SPINK
    A[2] = cos(snk1_theta*dtr) * cos(snk1_phi*dtr) ; // a(3) in MAD-SPINK

    if( coutdmp ){ //AUL:01MAR10
      std::cout << "\nGpuSnakeTransform " << m_name << ", turn = " << nturn << endl ;
      std::cout << "mu = " << snk1_mu << ", phi = " << snk1_phi << ", theta = " << snk1_theta << endl ;
      std::cout << "A[0] = " << A[0] << ", A[1] = " << A[1] << ", A[2] = " <<A[2] << endl ;
    }

    s_mat[0] = 1.00 - (A[1]*A[1] + A[2]*A[2])*cs ;
    s_mat[1] =      A[0]*A[1]*cs + A[2]*sn ;
    s_mat[2] =      A[0]*A[2]*cs - A[1]*sn ;
    
    s_mat[3] =      A[0]*A[1]*cs - A[2]*sn ;
    s_mat[4] = 1.00 - (A[0]*A[0] + A[2]*A[2])*cs ;
    s_mat[5] =      A[1]*A[2]*cs + A[0]*sn ;
    
    s_mat[6] =      A[0]*A[2]*cs + A[1]*sn ;
    s_mat[7] =      A[1]*A[2]*cs - A[0]*sn ;
    s_mat[8] = 1.00 - (A[0]*A[0] + A[1]*A[1])*cs ;

  } else if( m_name == "snake2" ) {
    // std::cout << "doing snake2 prop \n";
    precision cs = 1.00 -cos(snk2_mu*dtr) ; precision sn =  sin(snk2_mu*dtr) ;

    A[0] = cos(snk2_theta*dtr) * sin(snk2_phi*dtr) ; // a(1) in MAD-SPINk
    A[1] = sin(snk2_theta*dtr) ;                // a(2) in MAD-SPINK
    A[2] = cos(snk2_theta*dtr) * cos(snk2_phi*dtr) ; // a(3) in MAD-SPINK

    if( coutdmp ){ //AUL:01MAR10
      std::cout << "\nGpuSnakeTransform " << m_name << ", turn = " << nturn << endl ;
      std::cout << "mu = " << snk2_mu << ", phi = " << snk2_phi << ", theta = " << snk2_theta << endl ;
      std::cout << "A[0] = " << A[0] << ", A[1] = " << A[1] << ", A[2] = " <<A[2] << endl ;
    }

    s_mat[0] = 1.00 - (A[1]*A[1] + A[2]*A[2])*cs ;
    s_mat[1] =      A[0]*A[1]*cs + A[2]*sn ;
    s_mat[2] =      A[0]*A[2]*cs - A[1]*sn ;
    
    s_mat[3] =      A[0]*A[1]*cs - A[2]*sn ;
    s_mat[4] = 1.00 - (A[0]*A[0] + A[2]*A[2])*cs ;
    s_mat[5] =      A[1]*A[2]*cs + A[0]*sn ;
      
    s_mat[6] =      A[0]*A[2]*cs + A[1]*sn ;
    s_mat[7] =      A[1]*A[2]*cs - A[0]*sn ;
    s_mat[8] = 1.00 - (A[0]*A[0] + A[1]*A[1])*cs ;

  } else { //initialize spin matrix at the beginning of a turn
      /*if( nturn == 1 ) //AUL:01MAR10
	{} */
      OTs_mat[0][0] = OTs_mat[1][1] = OTs_mat[2][2] = 1.00 ;
      OTs_mat[0][1] = OTs_mat[0][2] = OTs_mat[1][0] = OTs_mat[1][2] = OTs_mat[2][0] = OTs_mat[2][1] = 0.00 ;
      
      if( coutdmp )//AUL:01MAR10
        {
	  std::cout << "\nSpin matrix initialize at " << m_name << ", turn = " << nturn << endl;
	  std::cout << "OT spin matrix" << endl ;
	  std::cout << OTs_mat[0][0] << "  " << OTs_mat[0][1] << "  " << OTs_mat[0][2] << endl ;
	  std::cout << OTs_mat[1][0] << "  " << OTs_mat[1][1] << "  " << OTs_mat[1][2] << endl ;
	  std::cout << OTs_mat[2][0] << "  " << OTs_mat[2][1] << "  " << OTs_mat[2][2] << endl ;
	}
     
  }


     
  //HANDLE_ERROR( cudaMemcpyToSymbol(s_matd,s_mat, sizeof(s_mat)));
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // std::cout << "before snake GPU push \n";
    // readPart(bunch);
gpu3dmatrix<<<blocksPerGrid, threadsPerBlock>>>(s_mat[0],s_mat[1],s_mat[2],s_mat[3],s_mat[4],s_mat[5],s_mat[6],s_mat[7],s_mat[8],N);
//std::cout << "after snake GPU push \n";
// readPart(bunch);
 
}








void SPINK::GpuTracker::loadPart(PAC::Bunch& bunch)
{
  
    PAC::BeamAttributes& ba = bunch.getBeamAttributes();
    precision e0    =   (precision) ba.getEnergy();
    precision m0    =  (precision) ba.getMass();
    precision GG    =  (precision) ba.getG();
    precision q           =  (precision)  ba.getCharge();
    precision p0 = sqrt(e0*e0 - m0*m0);
    precision gam = e0/m0;
    precision v0byc = p0/e0;
    precision Energy[PARTICLES],v0byc_c[PARTICLES],p0_c[PARTICLES];
    int N = bunch.size();
    precision dtr_h = atan(1.00)/45.00;
    //  cudaMemcpyToSymbol(p0_d,&p0,sizeof(precision));
    // cudaMemcpyToSymbol(Energy_d,&e0,sizeof(precision));
    cudaMemcpyToSymbol(GG_d,&GG,sizeof(precision));
    cudaMemcpyToSymbol(m0_d,&m0,sizeof(precision));
    cudaMemcpyToSymbol(q_d,&q,sizeof(precision));
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



  for(int ip = 0; ip <PARTICLES; ip++){
    pos[ip].sx = pos[ip].sy = pos[ip].sz = 0.00;
    pos[ip].x = pos[ip].y= pos[ip].ct = 0.00;
    pos[ip].px = pos[ip].py = pos[ip].de = 0.00;
    Energy[ip] = e0; p0_c[ip] = p0; v0byc_c[ip] = v0byc; 
}


 for(int ip = 0; ip < N; ip++) {
      PAC::Position& part = bunch[ip].getPosition();
      pos[ip].sx = (precision)  bunch[ip].getSpin()->getSX();
      pos[ip].sy=  (precision) bunch[ip].getSpin()->getSY();
      pos[ip].sz =  (precision) bunch[ip].getSpin()->getSZ();

  
        
      pos[ip].x = (precision)  part.getX(); pos[ip].px=  (precision)  part.getPX(); pos[ip].y= (precision)  part.getY();
      pos[ip].py = (precision)  part.getPY(); pos[ip].ct = (precision)  part.getCT(); pos[ip].de=  (precision) part.getDE();
   
      
    }
 // std::cout << "before sending to Gpu \n";
  cudaMemcpyToSymbol(pos_d,pos, sizeof(pos));
  cudaMemcpyToSymbol(tmp_d,pos, sizeof(pos));
  cudaMemcpyToSymbol(Energy_d,Energy,sizeof(Energy));
  cudaMemcpyToSymbol(p0_d,p0_c,sizeof(p0_c));
  cudaMemcpyToSymbol(v0byc_d,v0byc_c,sizeof(v0byc_c));
 // std::cout << "after sending to Gpu \n";


}

void SPINK::GpuTracker::readPart(PAC::Bunch& bunch,int printall)
{ int N = bunch.size();

 PAC::BeamAttributes& ba = bunch.getBeamAttributes();
  precision e0 = (precision)  ba.getEnergy(), m0 = (precision)  ba.getMass();
  precision gam ; //= e0/m0;
  precision Energy[PARTICLES];
   precision GG    =  (precision) ba.getG();
   // precision Ggam  = gam*GG; 
   precision SxAvg =0.00, SyAvg=0.00, SzAvg=0.00;
 int count =0;
  cudaMemcpyFromSymbol(Energy,Energy_d, sizeof(Energy));
    // cudaMemcpyFromSymbol(v0byc,v0byc_d,sizeof(v0byc));
  gam = Energy[0]/m0;
  precision Ggam  = gam*GG; 
//vec6D output[PARTICLES];
  cudaMemcpyFromSymbol(pos,pos_d, sizeof(pos));
 
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



}


SPINK::GpuTrackerRegister::GpuTrackerRegister()
{
  UAL::PropagatorNodePtr dipolePtr(new SPINK::GpuTracker());
  UAL::PropagatorFactory::getInstance().add("SPINK::GpuTracker", dipolePtr);
}

static SPINK::GpuTrackerRegister theSpinkGpuTrackerRegister;



