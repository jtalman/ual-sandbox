#ifndef gpu_kernels
#define gpu_kernels

#include <stdio.h>

__global__  void
Copygpu(){
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



__global__  void
passDriftgpu(precision rlipl, precision v0byc,int N, int size){

    int i = blockDim.x*blockIdx.x + threadIdx.x;
    precision X_out,Y_out,rvbyc,p1,p2,p4,CT_out;
   
         X_out = pos_d[i].x + (rlipl*tmp_d[i].px);
         Y_out = pos_d[i].y + (rlipl*tmp_d[i].py);
         if(size > 5){
         rvbyc = tmp_d[i].de;

         p2 = tmp_d[i].py*tmp_d[i].py;
         p1 = tmp_d[i].px*tmp_d[i].px + p2;
         p4 = (sqrt(1.00 + p1) + 1.00)/2.00;
         p1 = p1/p4;
           
         p1 = p1*rvbyc;
         p1 = p1*rlipl/2.00;
       
         p2 = 1.00/v0byc;
         p2 = p2 -rvbyc;
         p2 = p2*rlipl;


         CT_out = pos_d[i].ct - p1;
         CT_out = CT_out + p2;

        pos_d[i].ct = CT_out;

}

          pos_d[i].x = X_out; pos_d[i].y = Y_out; 

  return;
}


__global__  void
makeVelocitygpu(precision v0byc,int N){

      int i = blockDim.x*blockIdx.x + threadIdx.x;
      precision t0,t1,tmp_px,tmp_py;
    
      t0  = 1.00;
      t1  = pos_d[i].de;
      t1  = t1 + 2.00/v0byc;
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


__global__ void
makeRVgpu(precision v0byc, precision e0, precision p0, precision m0, int N){

    int i = blockDim.x*blockIdx.x + threadIdx.x;
    precision e, p2, rv;
   
    e = e0;
    e = e + p0*pos_d[i].de;
    
    p2 = e;
    p2 = p2*e;
    p2 = p2 - m0*m0;
    p2 = sqrt(p2);

    rv = e;
    rv = rv/p2;
    tmp_d[i].de = rv;


}

__global__  void
applyMltKickgpu( precision mlt0,precision mlt1, precision mlt2, precision mlt3, precision mlt4, precision mlt5, precision dx, precision dy, precision rkicks, int N, int size)
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
    
      t0 = x*px;
      t0 -= y*py - mlt4;
      py = x*py;
      py += y*px + mlt5;
      px = t0;   


      t0 = x*px;
      t0 -= y*py - mlt2;
      py = x*py;
      py += y*px + mlt3;
      px = t0;

      t0 = x*px;
      t0 -= y*py - mlt0;
      py = x*py;
      py += y*px + mlt1;
      px = t0;
         




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


__global__ void
applyThinBendKickgpu(precision v0byc, precision m_l, precision kl1,precision angle, precision btw01, precision btw00, precision atw01, precision atw00, precision dx, precision dy, precision rkicks, int N)
{  
  
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  precision x, y, px, py, PX_out, PY_out, CT_out;
   
  x = pos_d[i].x; //  xdif
  y = pos_d[i].y; //  ydif

  px = 0.00;
  py = 0.00;

    x = x - dx;
    y = y - dy;
    
 if(m_l){
  precision rho = m_l/angle;

    px -= x*btw01;
    px -= btw00;
    px += y*y*kl1*rkicks/rho/2.0;
   
    py += y*atw01;
    py += atw00;

    // px *= -1;

  PX_out = pos_d[i].px + px;
  PY_out = pos_d[i].py + py; 
} else {
     precision factor = angle/v0byc;
     px += factor*pos_d[i].de;
     CT_out = pos_d[i].ct - factor*pos_d[i].x;
     pos_d[i].ct = CT_out;
    }

 PX_out = px + pos_d[i].px;
 PY_out = py + pos_d[i].py;

  pos_d[i].px = PX_out; pos_d[i].py = PY_out;  
   
  return;         
}

__global__ void
passBendSlicegpu(precision cphpl, precision sphpl, precision tphpl, precision scrx,precision scrs, precision spxt, precision rlipl, precision v0byc, int N) {

 int i = blockDim.x*blockIdx.x + threadIdx.x;
  precision t0, t1, x,y, PX_out, CT_out, rvbyc;
  precision p0,p1,p2,p3,p4;
    
    t0  = ((cphpl)*pos_d[i].px);                                      
    t0 -= ((sphpl)*tmp_d[i].x);  
    PX_out = t0;

 // Transform coordinates into frame of multipole

 t0  = 1.00/(1.00 + ((tphpl)*tmp_d[i].px)); 
 t1  = (-scrx) + pos_d[i].x; 

   tmp_d[i].y = t0;
   if(rlipl){

  p0  = (pos_d[i].x + (spxt)*tmp_d[i].px);          
  p0 *= tmp_d[i].y;                                 
  p1  = p0 - pos_d[i].x;                               
  p0  = (tphpl)*((-scrx) + p0 ); 
  p3  = scrs - p0;                     
  p2  = tmp_d[i].py*p3;                              

  p4  = 0.00;                                      
  p1 *= p1;
  p2 *= p2;
  p3 *= p3;
  p4  = p1 + p2 + p3;
  p4  = sqrt(p4);                                            
  p4 += rlipl;     

  rvbyc = tmp_d[i].de;

  p1  = p1 + p2 + p0*p0;
  p1 -= scrx*scrx;             
  p1 -= (2.00*scrs)*p0;         

  p4 = p1/p4;
  p1  =  p4*rvbyc;             

  p2  = 1.00/v0byc;
  p2 -= rvbyc;
  p2 *= rlipl;

  CT_out = pos_d[i].ct - p1;               
  CT_out = CT_out + p2;               
   pos_d[i].ct = CT_out;

    }
 

 x  = t1;
 x += (scrs)*tmp_d[i].px;         
 x = ((1.00/(cphpl))*t0)*x;   
 y  = scrs;
 y -= tphpl*t1;                
 y *= tmp_d[i].py*t0;              
 y += pos_d[i].y;               


 pos_d[i].x = x;
 pos_d[i].y = y;

 pos_d[i].px = PX_out;
                              
 return;
}

__global__ void
gpuRFTracker(int N, precision lag, precision p0_old, 
precision e0_old, precision m0, precision m_l, precision v0byc_old, precision p0_new,
precision v0byc_new, precision h, precision q, precision V, precision revfreq_old, precision e0_new)
{
  precision pii = 3.1415926536;
  precision clite = 2.99792458e+8;
  int i = blockDim.x*blockIdx.x + threadIdx.x;
    precision X_out,Y_out,DE_out,CT_out,e_old,p_old,e_new,p_new,vbyc,de,phase;
    precision X_out2, Y_out2,CT_out2,vbyc_2;
    precision px_by_ps, py_by_ps, ps2_by_po2, t0, t0_2;
    precision dl2_by_lo2, l_by_lo,cdt_circ, cdt_vel;
    precision px_by_ps_2, py_by_ps_2, ps2_by_po2_2;
    precision dl2_by_lo2_2, l_by_lo_2,cdt_circ_2, cdt_vel_2;
   
    e_old = pos_d[i].de*p0_old + e0_old;
    p_old = sqrt(e_old*e_old - m0*m0);
    vbyc = p_old/e_old;
    
    //PassDrift part begin 
  ps2_by_po2 = 1.00 + (pos_d[i].de + 2.00/v0byc_old)*pos_d[i].de - pos_d[i].px*pos_d[i].px - pos_d[i].py*pos_d[i].py;
  t0 = 1.00/sqrt(ps2_by_po2);

  px_by_ps = pos_d[i].px*t0;
  py_by_ps = pos_d[i].py*t0;

  X_out = (m_l*px_by_ps/2.00) + pos_d[i].x;                
  Y_out = (m_l*py_by_ps/2.00) + pos_d[i].y;

  // Longitudinal part

  
  dl2_by_lo2  = px_by_ps*px_by_ps + py_by_ps*py_by_ps; // (L**2 - Lo**2)/Lo**2
  l_by_lo     = sqrt(1.00 + dl2_by_lo2);                 // L/Lo
  
  cdt_circ = dl2_by_lo2*m_l*0.50/(1 + l_by_lo)/vbyc;

  cdt_vel = m_l*0.50*(1.00/vbyc - 1.00/v0byc_old);

  // MAD longitudinal coordinate = -ct 

  CT_out = -cdt_vel - cdt_circ + pos_d[i].ct;
   
  //end of passDrift part


    phase = h*revfreq_old*(CT_out/clite);
    de = q*V*sin(2.00*pii*(lag - phase-0.5));
   
    e_new = e_old + de;
    DE_out = (e_new - e0_new)/p0_new ;
   
    //   printf("DEout = %e e_new = %e e0_new = %e p0_new = %e de = %e \n",DE_out,e_new,e0_new,p0_new,de);
    //   printf("pi = %e clight = %e phase = %e h = %e argsin = %e \n",pii,clite,phase,h, 2.00*pii*(lag-phase-0.5));

 
    p_new = sqrt(e_new*e_new - m0*m0);
    vbyc_2 = p_new/e_new;

    
     ps2_by_po2_2 = 1.00 + (DE_out + 2.00/v0byc_new)*DE_out - pos_d[i].px*pos_d[i].px - pos_d[i].py*pos_d[i].py;
  t0_2 = 1.00/sqrt(ps2_by_po2_2);

  px_by_ps_2 = pos_d[i].px*t0_2;
  py_by_ps_2 = pos_d[i].py*t0_2;

  X_out2 = (m_l*px_by_ps_2/2.00) + X_out;                
  Y_out2 = (m_l*py_by_ps_2/2.00) + Y_out;

  // Longitudinal part

  
  dl2_by_lo2_2  = px_by_ps_2*px_by_ps_2 + py_by_ps_2*py_by_ps_2; // (L**2 - Lo**2)/Lo**2
  l_by_lo_2     = sqrt(1.00 + dl2_by_lo2_2);                 // L/Lo
  
  cdt_circ_2 = dl2_by_lo2_2*m_l*0.50/(1 + l_by_lo_2)/vbyc_2;

  cdt_vel_2 = m_l*0.50*(1.00/vbyc_2 - 1.00/v0byc_new);

  // MAD longitudinal coordinate = -ct 

  CT_out2 = -cdt_vel_2 - cdt_circ_2 + CT_out;
     pos_d[i].x = X_out2; pos_d[i].y = Y_out2; pos_d[i].ct = CT_out2;
     pos_d[i].de = DE_out;



}

__global__ void
gpupropogateSpin(int N, precision ang,precision p0, precision e0, precision m0, precision h, precision length, precision k1l, precision k2l, precision k0l, precision kls0, precision v0byc, precision GG)
{
 int i = blockDim.x*blockIdx.x + threadIdx.x;
 precision e,p,gamma, KLx, KLy, vKL, fx, fy, fz, dt_by_ds;
 precision omega, pz, psp0 = 1;
 precision A0,A1,A2,cs,sn,sx1,sy1,sz1;
 precision s_mat00,s_mat01,s_mat02,s_mat10,s_mat20,s_mat11,s_mat22,s_mat21,s_mat12;
  
    e = pos_d[i].de*p0 + e0;
    p = sqrt(e*e - m0*m0);
    gamma = e/m0;
    psp0 -= pos_d[i].px*pos_d[i].px;
    psp0 -= pos_d[i].py*pos_d[i].py;

    psp0 += pos_d[i].de*pos_d[i].de;
    psp0 += (2.00/v0byc)*pos_d[i].de;

    pz = sqrt(psp0);
    
    


    KLx = k1l*pos_d[i].y + 2.00*k2l*pos_d[i].x*pos_d[i].y + kls0;
    KLy  = h*length + k1l*pos_d[i].x - k1l*pos_d[i].y*pos_d[i].y/2.0*h + k2l*(pos_d[i].x*pos_d[i].x - pos_d[i].y*pos_d[i].y) + k0l;  //VR added kls0 and k0l for kicker field effects.
    
    vKL = (pos_d[i].px*KLx + pos_d[i].py*KLy)/(p/p0);

    fx = (1.00 + GG*gamma)*KLx - GG*(gamma - 1.00)*vKL*pos_d[i].px/(p/p0);
    fy = (1.00 + GG*gamma)*KLy - GG*(gamma - 1.00)*vKL*pos_d[i].py/(p/p0);
    fz = -GG*(gamma - 1.00)*vKL*pz/(p/p0);

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


__global__ void gpu3dmatrix(precision s_mat0, precision s_mat1,precision s_mat2, precision s_mat3, precision s_mat4, precision s_mat5, precision s_mat6, precision s_mat7, precision s_mat8, int N)
{
 int i = blockDim.x*blockIdx.x + threadIdx.x;
 precision SX_out, SY_out, SZ_out;
   
    SX_out = pos_d[i].sx*s_mat0 + pos_d[i].sy*s_mat1 + pos_d[i].sz*s_mat2;
    SY_out = pos_d[i].sx*s_mat3 + pos_d[i].sy*s_mat4 + pos_d[i].sz*s_mat5;
    SZ_out = pos_d[i].sx*s_mat6 + pos_d[i].sy*s_mat7 + pos_d[i].sz*s_mat8;

    pos_d[i].sx = SX_out; pos_d[i].sy = SY_out; pos_d[i].sz = SZ_out;

return; 

}


#endif
