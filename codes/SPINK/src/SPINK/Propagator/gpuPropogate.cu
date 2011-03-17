#ifndef gpu_propogate
#define gpu_propogate

#include <stdio.h>

__constant__  precision s_steps[] = {0.10, 4.00/15, 4.00/15, 4.00/15, 0.10};



__device__  void CopyGpuP(int N){
  precision tmp_x,tmp_px,tmp_y,tmp_py,tmp_ct, tmp_de;
int i = blockDim.x*blockIdx.x + threadIdx.x;
 tmp_x = pos_d[i].x;
 tmp_px = pos_d[i].px;
 tmp_y = pos_d[i].y;
 tmp_py = pos_d[i].py;
 tmp_ct = pos_d[i].ct;
 tmp_de = pos_d[i].de;

 tmp_d[i].x = tmp_x;
 tmp_d[i].px = tmp_px;
 tmp_d[i].y = tmp_y;
 tmp_d[i].py = tmp_py;
 tmp_d[i].ct = tmp_ct;
 tmp_d[i].de = tmp_de;
 return;
}

__device__  void
applyMltKickgpuP(int N, int j,  precision rkicks,  int position, precision dx, precision dy)
{  
  
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  precision t0, x, y, px, py, PX_out, PY_out;
  // double kl, ktl;
  //   mlt[0] = mlt0; mlt[1]=mlt1; mlt[2]=mlt2; mlt[3]=mlt3;
  // mlt[4] =0.0; mlt[5]=0.0;mlt[6]=0.0; mlt[8]=0.0;
  //  mlt[9]=0.0;
  //  int index = size;
    
 
  x = pos_d[i].x; //  xdif
  y = pos_d[i].y; //  ydif

  px = 0.00;
  py = 0.00;
  t0 = 0.00;
  //  ktl = 0;
  // kl = 0 ;

  x = x - dx;
  y = y - dy;
 
    //   printf( " x = %f , y = %f \n", x,y);
  if(position == 0) {
      t0 = x*px;
      t0 -= y*py - rhic_d[j].entryMlt[4];
      py = x*py;
      py += y*px + rhic_d[j].entryMlt[5];
      px = t0;   


      t0 = x*px;
      t0 -= y*py - rhic_d[j].entryMlt[2];
      py = x*py;
      py += y*px + rhic_d[j].entryMlt[3];
      px = t0;

      t0 = x*px;
      t0 -= y*py - rhic_d[j].entryMlt[0];
      py = x*py;
      py += y*px + rhic_d[j].entryMlt[1];
      px = t0;
         
  }

if(position == 2) {
      t0 = x*px;
      t0 -= y*py - rhic_d[j].exitMlt[4];
      py = x*py;
      py += y*px + rhic_d[j].exitMlt[5];
      px = t0;   


      t0 = x*px;
      t0 -= y*py - rhic_d[j].exitMlt[2];
      py = x*py;
      py += y*px + rhic_d[j].exitMlt[3];
      px = t0;

      t0 = x*px;
      t0 -= y*py - rhic_d[j].exitMlt[0];
      py = x*py;
      py += y*px + rhic_d[j].exitMlt[1];
      px = t0;
         
  }

if(position == 1) {
      t0 = x*px;
      t0 -= y*py - rhic_d[j].mlt[4];
      py = x*py;
      py += y*px + rhic_d[j].mlt[5];
      px = t0;   


      t0 = x*px;
      t0 -= y*py - rhic_d[j].mlt[2];
      py = x*py;
      py += y*px + rhic_d[j].mlt[3];
      px = t0;

      t0 = x*px;
      t0 -= y*py - rhic_d[j].mlt[0];
      py = x*py;
      py += y*px + rhic_d[j].mlt[1];
      px = t0;
         
  }





      //    if( index > 0){
      // do {
      // ktl = mlt[--index];
      // kl  = mlt[--index];
      //   t0  = x*px;
      //  t0 -= y*py - kl;
      //  py  = x*py;
      //  py += y*px + ktl;    
      //  px  = t0;
      //        } while ( index > 0 ) ;
      // }
      px *= rkicks;
      py *= rkicks;
     
  

  px *= -1.00;

  PX_out = pos_d[i].px + px;
  PY_out = pos_d[i].py + py; 

  pos_d[i].px = PX_out; pos_d[i].py = PY_out;  
  

}

__device__  void
makeVelocitygpuP(int N){

      int i = blockDim.x*blockIdx.x + threadIdx.x;
      precision t0,t1,tmp_px,tmp_py;
    
      t0  = 1.00;
      t1  = pos_d[i].de;
      t1  = t1 + 2.00/v0byc_d;
      t1 = pos_d[i].de*t1;
      t0 = t1+t0;


      t1  = pos_d[i].px;
      t1  = t1*t1;
      t0  = t0-t1;

      t1  = pos_d[i].py;
      t1  = t1*t1;
      t0  = t0 - t1;

      t0  = sqrt(t0);
  // printf(" i =  %i pos_d[i].x = %f \n",i, pos_d[i].x);
  //  printf("tmp_d = %f  t0 = %f \n", tmp_d[i].x ,t0 );
      tmp_d[i].x   = t0;
     
  // printf("after it \n");

      t0  = 1.00/t0;
      tmp_px  = pos_d[i].px*t0;    // vx/vs
      tmp_py  = pos_d[i].py*t0;    // vy/vs
      tmp_d[i].px = tmp_px;
      tmp_d[i].py = tmp_py;

  return;
}

__device__ void
makeRVgpuP(int N){

    int i = blockDim.x*blockIdx.x + threadIdx.x;
    precision e, p2, rv;
    //  precision p0 = sqrt(Energy_d*Energy_d - m0_d*m0_d);
    e = Energy_d;
    e = e + p0_d*pos_d[i].de;
    
    p2 = e;
    p2 = p2*e;
    p2 = p2 - m0_d*m0_d;
    p2 = sqrt(p2);

    rv = e;
    rv = rv/p2;
    tmp_d[i].de = rv;


}

__device__ void
passBendSlicegpuP(int N, int j, int slice ) {

 int i = blockDim.x*blockIdx.x + threadIdx.x;
  precision t0, t1, x,y, PX_out, CT_out, rvbyc;
  precision p0,p1,p2,p3,p4;
    
    t0  = ((rhic_d[j].cphpl[slice])*pos_d[i].px);                                      
    t0 -= ((rhic_d[j].sphpl[slice])*tmp_d[i].x);  
    PX_out = t0;

 // Transform coordinates into frame of multipole

 t0  = 1.00/(1.00 + ((rhic_d[j].tphpl[slice])*tmp_d[i].px)); 
 t1  = (-rhic_d[j].scrx[slice]) + pos_d[i].x; 

   tmp_d[i].y = t0;
   if(rhic_d[j].rlipl[slice]){

  p0  = (pos_d[i].x + (rhic_d[j].spxt[slice])*tmp_d[i].px);          
  p0 *= tmp_d[i].y;                                 
  p1  = p0 - pos_d[i].x;                               
  p0  = (rhic_d[j].tphpl[slice])*((-rhic_d[j].scrx[slice]) + p0 ); 
  p3  = rhic_d[j].scrs[slice] - p0;                     
  p2  = tmp_d[i].py*p3;                              

  p4  = 0.00;                                      
  p1 *= p1;
  p2 *= p2;
  p3 *= p3;
  p4  = p1 + p2 + p3;
  p4  = sqrt(p4);                                            
  p4 += rhic_d[j].rlipl[slice];     

  rvbyc = tmp_d[i].de;

  p1  = p1 + p2 + p0*p0;
  p1 -= rhic_d[j].scrx[slice]*rhic_d[j].scrx[slice];             
  p1 -= (2.00*rhic_d[j].scrs[slice])*p0;         

  p4 = p1/p4;
  p1  =  p4*rvbyc;             

  p2  = 1.00/v0byc_d;
  p2 -= rvbyc;
  p2 *= rhic_d[j].rlipl[slice];

  CT_out = pos_d[i].ct - p1;               
  CT_out = CT_out + p2;               
   pos_d[i].ct = CT_out;

    }
 

 x  = t1;
 x += (rhic_d[j].scrs[slice])*tmp_d[i].px;         
 x = ((1.00/(rhic_d[j].cphpl[slice]))*t0)*x;   
 y  = rhic_d[j].scrs[slice];
 y -= rhic_d[j].tphpl[slice]*t1;                
 y *= tmp_d[i].py*t0;              
 y += pos_d[i].y;               


 pos_d[i].x = x;
 pos_d[i].y = y;

 pos_d[i].px = PX_out;
                              
 return;
}


__device__ void
applyThinBendKickgpuP(int N, int j,precision rkicks, precision dx, precision dy)
{  
  
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  precision x, y, px, py, PX_out, PY_out, CT_out;
   
  x = pos_d[i].x; //  xdif
  y = pos_d[i].y; //  ydif

  px = 0.00;
  py = 0.00;

    x = x - dx;
    y = y - dy;
    
 if(rhic_d[j].m_l > 0){
  precision rho = rhic_d[j].m_l/rhic_d[j].angle;

    px -= x*rhic_d[j].btw01;
    px -= rhic_d[j].btw00;
    px += y*y*rhic_d[j].kl1*rkicks/rho/2.0;
   
    py += y*rhic_d[j].atw01;
    py += rhic_d[j].atw00;

    // px *= -1;

  PX_out = pos_d[i].px + px;
  PY_out = pos_d[i].py + py; 
} else {
     precision factor = rhic_d[j].angle/v0byc_d;
     px += factor*pos_d[i].de;
     CT_out = pos_d[i].ct - factor*pos_d[i].x;
     pos_d[i].ct = CT_out;
    }

 PX_out = px + pos_d[i].px;
 PY_out = py + pos_d[i].py;

  pos_d[i].px = PX_out; pos_d[i].py = PY_out;  
   
  return;         
}

__device__  void
passDriftgpuP(int N, precision rlipl) {

    int i = blockDim.x*blockIdx.x + threadIdx.x;
    precision X_out,Y_out,rvbyc,p1,p2,p4,CT_out;
   
         X_out = pos_d[i].x + (rlipl*tmp_d[i].px);
         Y_out = pos_d[i].y + (rlipl*tmp_d[i].py);
        
         rvbyc = tmp_d[i].de;

         p2 = tmp_d[i].py*tmp_d[i].py;
         p1 = tmp_d[i].px*tmp_d[i].px + p2;
         p4 = (sqrt(1.00 + p1) + 1.00)/2.00;
         p1 = p1/p4;
           
         p1 = p1*rvbyc;
         p1 = p1*rlipl/2.00;
       
         p2 = 1.00/v0byc_d;
         p2 = p2 -rvbyc;
         p2 = p2*rlipl;


         CT_out = pos_d[i].ct - p1;
         CT_out = CT_out + p2;

        pos_d[i].ct = CT_out;
        pos_d[i].x = X_out; pos_d[i].y = Y_out; 

  return;

}

__device__ void RFProp(int N, int j)
{
   int i = blockDim.x*blockIdx.x + threadIdx.x;
precision de0 ,e0_new, p0_new ,v0byc_new ,revfreq_old ;
precision p0_old ,e0_old ,v0byc_old ;
precision X_out,Y_out,DE_out,CT_out,e_old,p_old,e_new,p_new,vbyc,de,phase;
    precision X_out2, Y_out2,CT_out2,vbyc_2;
    precision px_by_ps, py_by_ps, ps2_by_po2, t0, t0_2;
    precision dl2_by_lo2, l_by_lo,cdt_circ, cdt_vel;
    precision px_by_ps_2, py_by_ps_2, ps2_by_po2_2;
    precision dl2_by_lo2_2, l_by_lo_2,cdt_circ_2, cdt_vel_2;
   
     de0       = q_d*V_d*sin(2*PI_d*lag_d);
    e0_new    = Energy_d + de0;
  p0_new    = sqrt(e0_new*e0_new - m0_d*m0_d);
  v0byc_new = p0_new/e0_new;
  revfreq_old = v0byc_d*clite_d/circ_d ;
  p0_old = p0_d ;
  v0byc_old = v0byc_d;
  Energy_d = e0_new;
  v0byc_d = v0byc_new;
  e0_old = Energy_d;

  p0_d = p0_new;


    //printf(" before RF part = %f %f %f %f %f %f \n",pos_d[0].x,pos_d[0].px,pos_d[0].y,pos_d[0].py, pos_d[0].ct,pos_d[0].de);

    e_old = pos_d[i].de*p0_old + e0_old;
    p_old = sqrt(e_old*e_old - m0_d*m0_d);
    vbyc = p_old/e_old;
    
    //PassDrift part begin 
  ps2_by_po2 = 1.00 + (pos_d[i].de + 2.00/v0byc_old)*pos_d[i].de - pos_d[i].px*pos_d[i].px - pos_d[i].py*pos_d[i].py;
  t0 = 1.00/sqrt(ps2_by_po2);

  px_by_ps = pos_d[i].px*t0;
  py_by_ps = pos_d[i].py*t0;

  X_out = (rhic_d[j].m_l*px_by_ps/2.00) + pos_d[i].x;                
  Y_out = (rhic_d[j].m_l*py_by_ps/2.00) + pos_d[i].y;



  // Longitudinal part

  
  dl2_by_lo2  = px_by_ps*px_by_ps + py_by_ps*py_by_ps; // (L**2 - Lo**2)/Lo**2
  l_by_lo     = sqrt(1.00 + dl2_by_lo2);                 // L/Lo
  
  cdt_circ = dl2_by_lo2*rhic_d[j].m_l*0.50/(1 + l_by_lo)/vbyc;

  cdt_vel = rhic_d[j].m_l*0.50*(1.00/vbyc - 1.00/v0byc_old);

  // MAD longitudinal coordinate = -ct 

  CT_out = -cdt_vel - cdt_circ + pos_d[i].ct;
   
  //end of passDrift part
  //printf(" after RF pass drift 1st part = %f %f %f  \n",X_out,Y_out,CT_out);
  // printf( "V = %e h= %e  q = %e lag = %e revfreq_old = %e \n",V_d,h_d, q_d,lag_d,revfreq_old);

    phase = h_d*revfreq_old*(CT_out/clite_d);
    de = q_d*V_d*sin(2.00*PI_d*(lag_d - phase-0.5));
   
    e_new = e_old + de;
    DE_out = (e_new - e0_new)/p0_new ;
   
    //printf("DEout = %e e_new = %e e0_new = %e p0_new = %e de = %e phase = %e PI_d = %e \n",DE_out,e_new,e0_new,p0_new,de,phase,PI_d);
       //   printf("pi = %e clight = %e phase = %e h = %e argsin = %e \n",PI_d,clite_d,phase,h, 2.00*PI_d*(lag-phase-0.5));

 
    p_new = sqrt(e_new*e_new - m0_d*m0_d);
    vbyc_2 = p_new/e_new;

    
     ps2_by_po2_2 = 1.00 + (DE_out + 2.00/v0byc_new)*DE_out - pos_d[i].px*pos_d[i].px - pos_d[i].py*pos_d[i].py;
  t0_2 = 1.00/sqrt(ps2_by_po2_2);

  px_by_ps_2 = pos_d[i].px*t0_2;
  py_by_ps_2 = pos_d[i].py*t0_2;

  X_out2 = (rhic_d[j].m_l*px_by_ps_2/2.00) + X_out;                
  Y_out2 = (rhic_d[j].m_l*py_by_ps_2/2.00) + Y_out;

  // Longitudinal part

  
  dl2_by_lo2_2  = px_by_ps_2*px_by_ps_2 + py_by_ps_2*py_by_ps_2; // (L**2 - Lo**2)/Lo**2
  l_by_lo_2     = sqrt(1.00 + dl2_by_lo2_2);                 // L/Lo
  
  cdt_circ_2 = dl2_by_lo2_2*rhic_d[j].m_l*0.50/(1 + l_by_lo_2)/vbyc_2;

  cdt_vel_2 = rhic_d[j].m_l*0.50*(1.00/vbyc_2 - 1.00/v0byc_new);

  // MAD longitudinal coordinate = -ct 

  CT_out2 = -cdt_vel_2 - cdt_circ_2 + CT_out;
     pos_d[i].x = X_out2; pos_d[i].y = Y_out2; pos_d[i].ct = CT_out2;
     pos_d[i].de = DE_out;

  t0_d  += (rhic_d[j].m_l/v0byc_old + rhic_d[j].m_l/v0byc_new)/2./clite_d ;
  //printf(" after all RF part = %f %f %f %f %f %f \n",pos_d[0].x,pos_d[0].px,pos_d[0].y,pos_d[0].py, pos_d[0].ct,pos_d[0].de);
}



__device__ void BendProp(int N, int j)
{
  
  //printf("in BendProp \n");

  CopyGpuP(N);//

      
      
  if(rhic_d[j].entryMlt[0] != 1000){
      applyMltKickgpuP(N,j,1.0,0,0.,0.);//
  }
  makeVelocitygpuP(N);//
  makeRVgpuP(N);//
  
    
    if(rhic_d[j].m_ir == 0){
      passBendSlicegpuP(N,j,0);//
      
      if(rhic_d[j].mlt[0] != 1000){
	 applyMltKickgpuP(N,j,1.0,1,rhic_d[j].dx,rhic_d[j].dy); //
      }
      applyThinBendKickgpuP(N,j,1,rhic_d[j].dx,rhic_d[j].dy);//
      makeVelocitygpuP(N);//
      passBendSlicegpuP(N,j,1); //

	}else {
      precision rIr, rkicks; 
     rIr = 1./rhic_d[j].m_ir;
     rkicks = 0.25*rIr;
     int counter;
     counter = -1;
   for(int i = 0; i < rhic_d[j].m_ir; i++){
     for(int is = 1; is < 5; is++){
       counter++;
       passBendSlicegpuP(N,j,counter);//
       if(rhic_d[j].mlt[0] != 1000){
	  applyMltKickgpuP(N,j,rkicks,1,rhic_d[j].dx,rhic_d[j].dy); //
 }
       applyThinBendKickgpuP(N,j,rkicks,rhic_d[j].dx,rhic_d[j].dy);//
	makeVelocitygpuP(N);
     }
     counter++;
     passBendSlicegpuP(N,j,counter);
      makeVelocitygpuP(N);
   }
    }

    if(rhic_d[j].exitMlt[0] != 1000){
     
       applyMltKickgpuP(N,j,1,2,0.0,0.0); }
}



__device__ void MultProp(int N, int j)
{
  //  printf("MultProp \n");

  CopyGpuP(N);//
  if(rhic_d[j].entryMlt[0] != 1000 ){
    applyMltKickgpuP(N,j,1,0,0.0,0.0);//
  }
  makeVelocitygpuP(N);//
  makeRVgpuP(N);//

   if(rhic_d[j].m_ir == 0){
     passDriftgpuP(N,rhic_d[j].m_l/2.00);//
   if(rhic_d[j].mlt[0] != 1000) {
   applyMltKickgpuP(N,j,1.,1,rhic_d[j].dx,rhic_d[j].dy);//
    }
   makeVelocitygpuP(N);
   passDriftgpuP(N,rhic_d[j].m_l/2.00);
   if(rhic_d[j].exitMlt[0] != 1000) {
   applyMltKickgpuP(N,j,1.,2,0.0,0.0);//
}

   }

   precision rIr,rkicks ;
     rIr = 1.00/rhic_d[j].m_ir;
     rkicks = 0.25*rIr;
     int counter;
     counter = 0;
     for(int i = 0; i < rhic_d[j].m_ir; i++){
      for(int is = 0; is < 4; is++){
	counter++;
	passDriftgpuP(N,rhic_d[j].m_l*s_steps[is]*rIr);//
         if(rhic_d[j].mlt[0] != 1000) {
	   applyMltKickgpuP(N,j,rkicks,1,rhic_d[j].dx,rhic_d[j].dy); // 
        }
	 makeVelocitygpuP(N); //
      }
      counter++;
      passDriftgpuP(N,rhic_d[j].m_l*s_steps[4]*rIr); //
     }
      if(rhic_d[j].exitMlt[0] != 1000) {
    applyMltKickgpuP(N,j,1.,2,0.0,0.0); //
      }

       t0_d += rhic_d[j].m_l/v0byc_d/clite_d;
}



__device__ void DriftProp(int N, int j)
  {
    //printf("DriftProp \n");
    CopyGpuP(N);
    makeVelocitygpuP(N);
    makeRVgpuP(N);
    passDriftgpuP(N,rhic_d[j].m_l);
    t0_d += rhic_d[j].m_l/v0byc_d/clite_d;
  

   }







__device__ void
gpupropogateSpin(int N, precision ang, precision h, precision length, precision k1l, precision k2l, precision k0l, precision kls0)
{
 int i = blockDim.x*blockIdx.x + threadIdx.x;
 precision e,p,gamma, KLx, KLy, vKL, fx, fy, fz, dt_by_ds;
 precision omega, pz, psp0 = 1;
 precision A0,A1,A2,cs,sn,sx1,sy1,sz1;
 precision s_mat00,s_mat01,s_mat02,s_mat10,s_mat20,s_mat11,s_mat22,s_mat21,s_mat12;

 //printf("in gpupropogate Spin \n");
    e = pos_d[i].de*p0_d + Energy_d;
    p = sqrt(e*e - m0_d*m0_d);
    gamma = e/m0_d;
    psp0 -= pos_d[i].px*pos_d[i].px;
    psp0 -= pos_d[i].py*pos_d[i].py;

    psp0 += pos_d[i].de*pos_d[i].de;
    psp0 += (2.00/v0byc_d)*pos_d[i].de;

    pz = sqrt(psp0);
    
    


    KLx = k1l*pos_d[i].y + 2.00*k2l*pos_d[i].x*pos_d[i].y + kls0;
    KLy  = h*length + k1l*pos_d[i].x - k1l*pos_d[i].y*pos_d[i].y/2.0*h + k2l*(pos_d[i].x*pos_d[i].x - pos_d[i].y*pos_d[i].y) + k0l;  //VR added kls0 and k0l for kicker field effects.
    
    vKL = (pos_d[i].px*KLx + pos_d[i].py*KLy)/(p/p0_d);

    fx = (1.00 + GG_d*gamma)*KLx - GG_d*(gamma - 1.00)*vKL*pos_d[i].px/(p/p0_d);
    fy = (1.00 + GG_d*gamma)*KLy - GG_d*(gamma - 1.00)*vKL*pos_d[i].py/(p/p0_d);
    fz = -GG_d*(gamma - 1.00)*vKL*pz/(p/p0_d);

    dt_by_ds = (1.00 + h*pos_d[i].x)/pz;

    fx *= dt_by_ds;
    fy *= dt_by_ds;
    fz *= dt_by_ds;

    omega = sqrt(fx*fx + (fy - h*length)*(fy - h*length) + fz*fz);
    
     if( omega > 0.00 ) {
    
      cs = 1.00 - cos(omega); 
      sn = sin(omega);

    
    
      A0 = fx/omega;
      A1 = (fy - h*length)/omega;
      A2 = fz/omega;

    s_mat00 = 1.00 - (A1*A1 + A2*A2)*cs ;
    s_mat01 =      A0*A1*cs + A2*sn ;
    s_mat02 =      A0*A2*cs - A1*sn ;
    
    s_mat10 =      A0*A1*cs - A2*sn ;
    s_mat11 = 1.00 - (A0*A0 + A2*A2)*cs ;
    s_mat12 =      A1*A2*cs + A0*sn ;
    
    s_mat20 =      A0*A2*cs + A1*sn ;
    s_mat21 =      A1*A2*cs - A0*sn ;
    s_mat22 = 1.00 - (A0*A0 + A1*A1)*cs ;

  } else {
    s_mat00 = s_mat11 = s_mat22 = 1.00 ;
    s_mat01 = s_mat02 = s_mat10 = s_mat12 = s_mat20 = s_mat21 = 0.00 ;
  }

    sx1 = s_mat00*pos_d[i].sx + s_mat01*pos_d[i].sy + s_mat02*pos_d[i].sz;
    sy1 = s_mat10*pos_d[i].sx + s_mat11*pos_d[i].sy + s_mat12*pos_d[i].sz;
    sz1 = s_mat20*pos_d[i].sx + s_mat21*pos_d[i].sy + s_mat22*pos_d[i].sz;
  
    pos_d[i].sx = sx1; pos_d[i].sy = sy1; pos_d[i].sz = sz1;

   }

__device__ void gpu3dmatrixP(precision s_mat0, precision s_mat1,precision s_mat2, precision s_mat3, precision s_mat4, precision s_mat5, precision s_mat6, precision s_mat7, precision s_mat8, int N)
{
 int i = blockDim.x*blockIdx.x + threadIdx.x;
 precision SX_out, SY_out, SZ_out;
   
    SX_out = pos_d[i].sx*s_mat0 + pos_d[i].sy*s_mat1 + pos_d[i].sz*s_mat2;
    SY_out = pos_d[i].sx*s_mat3 + pos_d[i].sy*s_mat4 + pos_d[i].sz*s_mat5;
    SZ_out = pos_d[i].sx*s_mat6 + pos_d[i].sy*s_mat7 + pos_d[i].sz*s_mat8;

    pos_d[i].sx = SX_out; pos_d[i].sy = SY_out; pos_d[i].sz = SZ_out;

return; 

}
__device__ void SnakeProp(int N,int j)
{
  //printf("SnakeProp \n");
  precision dtr;
  dtr = atan(1.00)/45.00;
  

  precision A[3] ;
  precision s_mat[9] ;

 
  if( rhic_d[j].snake == 1) {
    //   std::cout << "doing snake1 prop \n";
    precision cs,sn;
   cs = 1.00 -cos(snk1_mu_d*dtr) ; sn =  sin(snk1_mu_d*dtr) ;

    A[0] = cos(snk1_theta_d*dtr) * sin(snk1_phi_d*dtr) ; // a(1) in MaD-SPINk
    A[1] = sin(snk1_theta_d*dtr) ;                // a(2) in MAD-SPINK
    A[2] = cos(snk1_theta_d*dtr) * cos(snk1_phi_d*dtr) ; // a(3) in MAD-SPINK

    s_mat[0] = 1.00 - (A[1]*A[1] + A[2]*A[2])*cs ;
    s_mat[1] =      A[0]*A[1]*cs + A[2]*sn ;
    s_mat[2] =      A[0]*A[2]*cs - A[1]*sn ;
    
    s_mat[3] =      A[0]*A[1]*cs - A[2]*sn ;
    s_mat[4] = 1.00 - (A[0]*A[0] + A[2]*A[2])*cs ;
    s_mat[5] =      A[1]*A[2]*cs + A[0]*sn ;
    
    s_mat[6] =      A[0]*A[2]*cs + A[1]*sn ;
    s_mat[7] =      A[1]*A[2]*cs - A[0]*sn ;
    s_mat[8] = 1.00 - (A[0]*A[0] + A[1]*A[1])*cs ;

  } else if( rhic_d[j].snake == 2 ) {
    // std::cout << "doing snake2 prop \n";
    precision cs,sn;
    cs = 1.00 -cos(snk2_mu_d*dtr) ; sn =  sin(snk2_mu_d*dtr) ;

    A[0] = cos(snk2_theta_d*dtr) * sin(snk2_phi_d*dtr) ; // a(1) in MAD-SPINk
    A[1] = sin(snk2_theta_d*dtr) ;                // a(2) in MAD-SPINK
    A[2] = cos(snk2_theta_d*dtr) * cos(snk2_phi_d*dtr) ; // a(3) in MAD-SPINK


    s_mat[0] = 1.00 - (A[1]*A[1] + A[2]*A[2])*cs ;
    s_mat[1] =      A[0]*A[1]*cs + A[2]*sn ;
    s_mat[2] =      A[0]*A[2]*cs - A[1]*sn ;
    
    s_mat[3] =      A[0]*A[1]*cs - A[2]*sn ;
    s_mat[4] = 1.00 - (A[0]*A[0] + A[2]*A[2])*cs ;
    s_mat[5] =      A[1]*A[2]*cs + A[0]*sn ;
      
    s_mat[6] =      A[0]*A[2]*cs + A[1]*sn ;
    s_mat[7] =      A[1]*A[2]*cs - A[0]*sn ;
    s_mat[8] = 1.00 - (A[0]*A[0] + A[1]*A[1])*cs ;

  } else { 

    return;
     
  }


     
  
gpu3dmatrixP(s_mat[0],s_mat[1],s_mat[2],s_mat[3],s_mat[4],s_mat[5],s_mat[6],s_mat[7],s_mat[8],N);
 
}
__device__ void propagateSpin(int N, int j)
{
  //printf("propagateSpin \n"); 
 int ns = 1;
  if(rhic_d[j].ns > 0) ns = 4*rhic_d[j].ns;

  precision length;
  length = rhic_d[j].l/ns;

  precision ang,h; 
  ang = 0.00; h = 0.00;
  if(rhic_d[j].bend > 0)  {
      ang    = rhic_d[j].bend/ns;
// VR added to avoid div by zero Dec22 2010
     if(length > 1e-20)
      h      = ang/length;
  }

  precision k1l,k2l,k0l,kls0;
  k1l = 0.00; k2l = 0.00;
  k0l = 0.00; kls0 = 0.00; // VR added to handle hkicker and vkicker spin effects Dec22 2010
  if(rhic_d[j].mlt[0] != 1000){
    if(rhic_d[j].order == 0){
      k0l = rhic_d[j].k0l/ns; kls0 = rhic_d[j].kls0/ns ;} // VR added to handle hkicker and vkicker spin effects Dec22 2010
    if(rhic_d[j].order > 0) k1l   =   rhic_d[j].k1l/ns;
    if(rhic_d[j].order > 1) k2l   =   rhic_d[j].k2l/ns;
  }

 if(length != 0 || k1l != 0 || k2l != 0) {
   gpupropogateSpin(N, ang, h, length, k1l, k2l, k0l,kls0);
    

 } 
    

}

__global__ void gpuPropagate(int N){

  //printf(" in gpuProp  \n");
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  precision length,ang;
   length=0.00; ang = 0.00;
 // precision p0_d = sqrt(Energy_d*Energy_d - m0_d*m0_d);
 // precision gam_d = Energy_d/m0_d;
 precision v;
 v = p0_d/gam_d/m0_d/clite_d;
 // precision v0byc_d = p0_d/Energy;
 // looping over lattice elements 
for(int j = 0 ; j < 1845 ; j++)
   {
     //  printf( "starting element num = %i on thread = %i  \n",j,i);
     // printf(" i =  %i pos_d[0].px = %f \n",i, pos_d[0].px);
     // printf("length  = %f at element j = %i \n", rhic_d[j].l,j); 
     //       printf("copied name = %s \n", rhic_d[j].m_name);
     //     printf("bend = %f mlt = %f\n",rhic_d[j].bend,rhic_d[j].mlt[0]);
      //continue ;
     //  printf("part = %f %f %f %f %f %f \n",pos_d[0].x,pos_d[0].px,pos_d[0].y,pos_d[0].py, pos_d[0].ct,pos_d[0].de);
    
       ang = fabs(rhic_d[j].bend); 
       length = rhic_d[j].l;
     if(rhic_d[j].ns == 0) {
       //  printf(" in simple element \n");
       length   /= 2.00;
       if(rhic_d[j].mlt[0] != 1000){ for(int k=0;k<10;k++) rhic_d[j].mlt[k] /= 2.00;}  
       if(rhic_d[j].rfcav == 1 ){
	 // printf(" found RF cavity \n");
	 RFProp(N,j);
	 continue;
       }
          

	 if(ang > 0.00){
	   //	     printf( "going to BendProp particles are = %f, %f , %f , %f \n",pos_d[0].x,pos_d[0].px,pos_d[0].y, pos_d[0].py );
	   BendProp(N,j);
	   //  printf( "after BendProp particles are = %f, %f , %f , %f \n",pos_d[0].x,pos_d[0].px,pos_d[0].y, pos_d[0].py );
	 }else if(rhic_d[j].mlt[0] != 1000) {
	   MultProp(N,j);
	 }else if(length > 0) {
	   DriftProp(N,j);
	 }

	 if(rhic_d[j].mlt[0] != 1000) { for(int k=0;k<10;k++) rhic_d[j].mlt[k] *= 2.00;}    
        t0_d += length/v;
     //    ba.setElapsedTime(t0);

	if(rhic_d[j].snake > 0 ){
  
	  SnakeProp(N,j);} else{
	  propagateSpin(N,j);}


	if(rhic_d[j].mlt[0] != 1000){ for(int k=0;k<10;k++) rhic_d[j].mlt[k] /= 2.00; } 
    if(ang > 0.00){
	
      BendProp(N,j);
	
    }else if(rhic_d[j].mlt[0] != 1000) {
	
      MultProp(N,j);
	
 } else if( length > 0){
   
      DriftProp(N,j);
      
}
   
    if(rhic_d[j].mlt[0] != 1000) { for(int k=0;k<10;k++) rhic_d[j].mlt[k] *= 2.00;}  

    t0_d += length/v;
    //ba.setElapsedTime(t0);
    //printf(" after part = %f %f %f %f %f %f \n",pos_d[0].x,pos_d[0].px,pos_d[0].y,pos_d[0].py, pos_d[0].ct,pos_d[0].de);
    continue;
     }

    
  //std::cout << " in complex element  ang = " << ang << " \n";
   
     int ns = rhic_d[j].ns*4;

   length /= 2*ns;

  for(int i=0; i < ns; i++) {

    if(rhic_d[j].mlt[0] != 1000) { for (int k= 0 ;k <10 ; k++) rhic_d[j].mlt[k] /= (2*ns);}          // kl, kt
   if(ang > 0.00){
	
     BendProp(N,j);

 }else if(rhic_d[j].mlt[0] != 1000) {

     MultProp(N,j);
	
 }  else if( length > 0){
     
     DriftProp(N,j);
      
   }
   
   if(rhic_d[j].mlt[0] != 1000){ for(int k=0;k <10;k++) rhic_d[j].mlt[k] *= (2*ns);}          // kl, kt
 
    t0_d += length/v;
    //    ba.setElapsedTime(t0);
    if(rhic_d[j].snake > 0){
     
      SnakeProp(N,j);} else{
      propagateSpin(N,j);}

    if(rhic_d[j].mlt[0] != 1000)  { for (int k= 0 ;k <10 ; k++) rhic_d[j].mlt[k] /= (2*ns);}             // kl, kt

    if(ang > 0.00){
	
      BendProp(N,j);

    }else if(rhic_d[j].mlt[0] != 1000) {

      MultProp(N,j);
	  
    } else if( length > 0){
     
      DriftProp(N,j);
      
   }
   
    if(rhic_d[j].mlt[0] != 1000) { for(int k=0;k <10;k++) rhic_d[j].mlt[k] *= (2*ns);}                 // kl, kt
 
    t0_d += length/v;
    //    ba.setElapsedTime(t0);

  }
       
  

  // printf(" after part = %f %f %f %f %f %f \n",pos_d[0].x,pos_d[0].px,pos_d[0].y,pos_d[0].py, pos_d[0].ct,pos_d[0].de);
   }

}












#endif
     
