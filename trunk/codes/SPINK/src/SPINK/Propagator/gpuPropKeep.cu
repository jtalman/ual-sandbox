#ifndef gpu_propogate
#define gpu_propogate

#include <stdio.h>

__constant__  precision s_steps[] = {0.10, 4.00/15, 4.00/15, 4.00/15, 0.10};
__constant__ precision m0_d, circ_d, GG_d, q_d;
__constant__ precision snk1_mu_d,snk1_theta_d,snk1_phi_d;
__constant__ precision snk2_mu_d,snk2_theta_d,snk2_phi_d;
__constant__ precision PI_d=3.1415926536, clite_d= 2.99792458e+8;
__constant__ precision V_d, lag_d, h_d,dtr,small = 1e-20;
__device__ precision Energy_d, v0byc_d, p0_d,gam_d, t0_d;


__device__  void
applyMltKickgpuP(int N, int j,  precision rkicks, int position, precision dx, precision dy, volatile precision x1,volatile precision &px1, volatile precision y1, volatile precision &py1)
{  
  precision t0, x,y, px, py, PX_out, PY_out;
  px = 0.00;
  py = 0.00;
  t0 = 0.00; 
  x = x1 - dx;
  y = y1 - dy;
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


      px *= rkicks;
      py *= rkicks;
     
  

  px *= -1.00;

  PX_out = px1 + px;
  PY_out = py1 + py; 

  px1 = PX_out; py1 = PY_out;  
  

}

__device__  void
makeVelocitygpuP(int N,volatile precision px, volatile precision py,volatile precision de, precision &xt, precision &pxt, precision &pyt){
      precision t0,t1,tmp_px,tmp_py;
    
      t0  = 1.00;
      t1  = de;
      t1  = t1 + 2.00/v0byc_d;
      t1 = de*t1;
      t0 = t1+t0;


      t1  = px;
      t1  = t1*t1;
      t0  = t0-t1;

      t1  = py;
      t1  = t1*t1;
      t0  = t0 - t1;

      t0  = sqrt(t0);
  
      xt   = t0;
     
  // printf("after it \n");

      t0  = 1.00/t0;
      tmp_px  = px*t0;    // vx/vs
      tmp_py  = py*t0;    // vy/vs
      pxt = tmp_px;
      pyt = tmp_py;

  return;
}


__device__ void
makeRVgpuP(int N, volatile precision de, precision &det){
    precision e, p2, rv;
    e = Energy_d;
    e = e + p0_d*de;
    
    p2 = e;
    p2 = p2*e;
    p2 = p2 - m0_d*m0_d;
    p2 = sqrt(p2);

    rv = e;
    rv = rv/p2;
    det = rv;


}

__device__ void
passBendSlicegpuP(int N, int j, int slice, volatile precision &x1,volatile precision &px1, volatile precision &y1, volatile precision py1, volatile precision &ct1, volatile precision de1, precision xt,precision pxt, precision yt, precision pyt, precision det) {
  precision t0, t1, x,y, PX_out, CT_out, rvbyc;
  precision p0,p1,p2,p3,p4;
    
    t0  = ((rhic_d[j].cphpl[slice])*px1);                                      
    t0 -= ((rhic_d[j].sphpl[slice])*xt);  
    PX_out = t0;

 // Transform coordinates into frame of multipole

 t0  = 1.00/(1.00 + ((rhic_d[j].tphpl[slice])*pxt)); 
 t1  = (-rhic_d[j].scrx[slice]) + x1; 

   yt = t0;
   if(rhic_d[j].rlipl[slice]){

  p0  = (x1 + (rhic_d[j].spxt[slice])*pxt);          
  p0 *= yt;                                 
  p1  = p0 - x1;                               
  p0  = (rhic_d[j].tphpl[slice])*((-rhic_d[j].scrx[slice]) + p0 ); 
  p3  = rhic_d[j].scrs[slice] - p0;                     
  p2  = pyt*p3;                              

  p4  = 0.00;                                      
  p1 *= p1;
  p2 *= p2;
  p3 *= p3;
  p4  = p1 + p2 + p3;
  p4  = sqrt(p4);                                            
  p4 += rhic_d[j].rlipl[slice];     

  rvbyc = det;

  p1  = p1 + p2 + p0*p0;
  p1 -= rhic_d[j].scrx[slice]*rhic_d[j].scrx[slice];             
  p1 -= (2.00*rhic_d[j].scrs[slice])*p0;         

  p4 = p1/p4;
  p1  =  p4*rvbyc;             

  p2  = 1.00/v0byc_d;
  p2 -= rvbyc;
  p2 *= rhic_d[j].rlipl[slice];

  CT_out = ct1 - p1;               
  CT_out = CT_out + p2;               
  ct1 = CT_out;

    }
 

 x  = t1;
 x += (rhic_d[j].scrs[slice])*pxt;         
 x = ((1.00/(rhic_d[j].cphpl[slice]))*t0)*x;   
 y  = rhic_d[j].scrs[slice];
 y -= rhic_d[j].tphpl[slice]*t1;                
 y *= pyt*t0;              
 y += y1;               


 x1 = x;
 y1 = y;

 px1 = PX_out;
                              
 return;
}


__device__ void
applyThinBendKickgpuP(int N, int j,precision rkicks, precision dx, precision dy,volatile precision x1,volatile precision &px1, volatile precision y1, volatile precision &py1, volatile precision &ct1, volatile precision de1)
{  
  precision x, y, px, py, PX_out, PY_out, CT_out;
   
  x = x1; //  xdif
  y = y1; //  ydif

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

  PX_out = px1 + px;
  PY_out = py1 + py; 
} else {
     precision factor = rhic_d[j].angle/v0byc_d;
     px += factor*de1;
     CT_out = ct1 - factor*x1;
     ct1 = CT_out;
    }

 PX_out = px + px1;
 PY_out = py + py1;

  px1 = PX_out; py1 = PY_out;  
   
  return;         
}

__device__  void
passDriftgpuP(int N, precision rlipl, volatile precision &x,  precision pxt, volatile precision &y, precision pyt, volatile precision &ct, precision det) {

    precision X_out,Y_out,rvbyc,p1,p2,p4,CT_out;
   
         X_out = x + (rlipl*pxt);
         Y_out = y + (rlipl*pyt);
        
         rvbyc = det;

         p2 = pyt*pyt;
         p1 = pxt*pxt + p2;
         p4 = (sqrt(1.00 + p1) + 1.00)/2.00;
         p1 = p1/p4;
           
         p1 = p1*rvbyc;
         p1 = p1*rlipl/2.00;
       
         p2 = 1.00/v0byc_d;
         p2 = p2 -rvbyc;
         p2 = p2*rlipl;


         CT_out = ct - p1;
         CT_out = CT_out + p2;

        ct = CT_out;
        x = X_out; y = Y_out; 

  return;

}

__device__ void RFProp(int N, int j,volatile precision &x1,volatile precision px1, volatile precision &y1, volatile precision py1, volatile precision &ct1, volatile precision &de1)
{
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

    e_old = de1*p0_old + e0_old;
    p_old = sqrt(e_old*e_old - m0_d*m0_d);
    vbyc = p_old/e_old;
    
    //PassDrift part begin 
  ps2_by_po2 = 1.00 + (de1 + 2.00/v0byc_old)*de1 - px1*px1 - py1*py1;
  t0 = 1.00/sqrt(ps2_by_po2);

  px_by_ps = px1*t0;
  py_by_ps = py1*t0;

  X_out = (rhic_d[j].m_l*px_by_ps/2.00) + x1;                
  Y_out = (rhic_d[j].m_l*py_by_ps/2.00) + y1;



  // Longitudinal part

  
  dl2_by_lo2  = px_by_ps*px_by_ps + py_by_ps*py_by_ps; // (L**2 - Lo**2)/Lo**2
  l_by_lo     = sqrt(1.00 + dl2_by_lo2);                 // L/Lo
  
  cdt_circ = dl2_by_lo2*rhic_d[j].m_l*0.50/(1 + l_by_lo)/vbyc;

  cdt_vel = rhic_d[j].m_l*0.50*(1.00/vbyc - 1.00/v0byc_old);

  // MAD longitudinal coordinate = -ct 

  CT_out = -cdt_vel - cdt_circ + ct1;
   
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

    
     ps2_by_po2_2 = 1.00 + (DE_out + 2.00/v0byc_new)*DE_out - px1*px1 - py1*py1;
  t0_2 = 1.00/sqrt(ps2_by_po2_2);

  px_by_ps_2 = px1*t0_2;
  py_by_ps_2 = py1*t0_2;

  X_out2 = (rhic_d[j].m_l*px_by_ps_2/2.00) + X_out;                
  Y_out2 = (rhic_d[j].m_l*py_by_ps_2/2.00) + Y_out;

  // Longitudinal part

  
  dl2_by_lo2_2  = px_by_ps_2*px_by_ps_2 + py_by_ps_2*py_by_ps_2; // (L**2 - Lo**2)/Lo**2
  l_by_lo_2     = sqrt(1.00 + dl2_by_lo2_2);                 // L/Lo
  
  cdt_circ_2 = dl2_by_lo2_2*rhic_d[j].m_l*0.50/(1 + l_by_lo_2)/vbyc_2;

  cdt_vel_2 = rhic_d[j].m_l*0.50*(1.00/vbyc_2 - 1.00/v0byc_new);

  // MAD longitudinal coordinate = -ct 

  CT_out2 = -cdt_vel_2 - cdt_circ_2 + CT_out;
     x1 = X_out2; y1 = Y_out2; ct1 = CT_out2;
     de1 = DE_out;

  t0_d  += (rhic_d[j].m_l/v0byc_old + rhic_d[j].m_l/v0byc_new)/2./clite_d ;
  //printf(" after all RF part = %f %f %f %f %f %f \n",pos_d[0].x,pos_d[0].px,pos_d[0].y,pos_d[0].py, pos_d[0].ct,pos_d[0].de);
}



__device__ void BendProp(int N, int j, volatile precision &x, volatile precision &px, volatile precision &y, volatile precision &py, volatile precision &ct, volatile precision &de)
{
  precision xt,pxt,yt,pyt,ctt,det,dx,dy;
  dx = rhic_d[j].dx; dy = rhic_d[j].dy;
   xt = x;
 pxt = px;
 yt = y;
 pyt = py;
 ctt = ct;
 det = de;
      
      
  if(rhic_d[j].entryMlt[0] < 1000){
    applyMltKickgpuP(N,j,1.0,0,0.,0. ,x, px, y, py);//
  }
  makeVelocitygpuP(N,px, py,de,xt,pxt, pyt);//
  makeRVgpuP(N, de, det);//
  
    
    if(rhic_d[j].m_ir == 0){
      passBendSlicegpuP(N,j,0, x, px, y, py, ct,de, xt, pxt, yt, pyt, det);//
      
      if(rhic_d[j].mlt[0] < 1000){
	 applyMltKickgpuP(N,j,1.0,1,dx,dy,x, px, y, py); //
      }
      applyThinBendKickgpuP(N,j,1,dx,dy, x,px,y,py,ct, de);//
      makeVelocitygpuP(N,px, py,de,xt,pxt, pyt);//
      passBendSlicegpuP(N,j,1, x, px, y, py, ct,de, xt, pxt, yt, pyt, det); //

	}else {
      precision rIr, rkicks;
   
     rIr = 1./rhic_d[j].m_ir;
     rkicks = 0.25*rIr;
     int counter;
     counter = -1;
   for(int i = 0; i < rhic_d[j].m_ir; i++){
     for(int is = 1; is < 5; is++){
       counter++;
       passBendSlicegpuP(N,j,counter, x, px, y, py, ct,de, xt, pxt, yt, pyt, det);//
       if(rhic_d[j].mlt[0] < 1000){
	 applyMltKickgpuP(N,j,rkicks,1,dx,dy,x, px, y, py); //
 }
       applyThinBendKickgpuP(N,j,rkicks,dx,dy,x, px, y, py,ct,de);//
	makeVelocitygpuP(N,px, py,de,xt,pxt, pyt);
     }
     counter++;
     passBendSlicegpuP(N,j,counter, x, px, y, py, ct,de, xt, pxt, yt, pyt, det);
      makeVelocitygpuP(N,px, py,de,xt,pxt, pyt);
   }
    }

    if(rhic_d[j].exitMlt[0] < 1000){
     
       applyMltKickgpuP(N,j,1,2,0.0,0.0,x, px, y, py); }
}



__device__ void MultProp(int N, int j, volatile precision &x, volatile precision &px, volatile precision &y, volatile precision &py, volatile precision &ct, volatile precision &de)
{
  //  printf("MultProp \n");
  precision xt,pxt,yt,pyt,ctt,det,dx,dy;
  dx = rhic_d[j].dx; dy = rhic_d[j].dy;;
  
   xt = x;
 pxt = px;
 yt = y;
 pyt = py;
 ctt = ct;
 det = de;
  
  if(rhic_d[j].entryMlt[0] < 1000 ){
    applyMltKickgpuP(N,j,1,0,0.0,0.0,x, px, y, py);//
  }
  makeVelocitygpuP(N,px, py,de,xt,pxt, pyt);//
  makeRVgpuP(N, de, det);//

   if(rhic_d[j].m_ir == 0){
     passDriftgpuP(N,rhic_d[j].m_l/2.00, x, pxt, y, pyt, ct, det);//
   if(rhic_d[j].mlt[0] < 1000) {
   applyMltKickgpuP(N,j,1.,1,dx,dy,x, px, y, py);//
    }
   makeVelocitygpuP(N,px, py,de,xt,pxt, pyt);
   passDriftgpuP(N,rhic_d[j].m_l/2.00, x, pxt, y, pyt, ct, det);
   if(rhic_d[j].exitMlt[0] < 1000) {
   applyMltKickgpuP(N,j,1.,2,0.0,0.0,x, px, y, py);//
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
	passDriftgpuP(N,rhic_d[j].m_l*s_steps[is]*rIr, x, pxt, y, pyt, ct, det);//
         if(rhic_d[j].mlt[0] < 1000) {
	   applyMltKickgpuP(N,j,rkicks,1,dx,dy,x, px, y, py); // 
        }
	 makeVelocitygpuP(N,px, py,de,xt,pxt, pyt); //
      }
      counter++;
      passDriftgpuP(N,rhic_d[j].m_l*s_steps[4]*rIr, x, pxt, y, pyt, ct, det); //
     }
      if(rhic_d[j].exitMlt[0] < 1000) {
    applyMltKickgpuP(N,j,1.,2,0.0,0.0,x, px, y, py); //
      }

       t0_d += rhic_d[j].m_l/v0byc_d/clite_d;
}



__device__ void DriftProp(int N, int j, volatile precision &x, volatile precision &px, volatile precision &y, volatile precision &py, volatile precision &ct, volatile precision &de)
  {
     precision xt,pxt,yt,pyt,ctt,det;
  
   xt = x;
 pxt = px;
 yt = y;
 pyt = py;
 ctt = ct;
 det = de;
   
    makeVelocitygpuP(N,px, py,de,xt,pxt, pyt);
    makeRVgpuP(N, de, det);
    passDriftgpuP(N,rhic_d[j].m_l, x, pxt, y, pyt, ct, det);
    t0_d += rhic_d[j].m_l/v0byc_d/clite_d;
  

   }







__device__ void
gpupropogateSpin(int N, precision ang, precision h, precision length, precision k1l, precision k2l, precision k0l, precision kls0,  volatile precision x, volatile precision px, volatile precision y, volatile precision py, volatile precision ct, volatile precision de, volatile precision &sx0, volatile precision &sy0, volatile precision &sz0)
{
 precision e,p,gamma, KLx, KLy, vKL, fx, fy, fz, dt_by_ds;
 precision omega, pz, psp0 = 1;
 precision A0,A1,A2,cs,sn,sx1,sy1,sz1;
 precision s_mat00,s_mat01,s_mat02,s_mat10,s_mat20,s_mat11,s_mat22,s_mat21,s_mat12;

 //printf("in gpupropogate Spin \n");
    e = de*p0_d + Energy_d;
    p = sqrt(e*e - m0_d*m0_d);
    gamma = e/m0_d;
    psp0 -= px*px;
    psp0 -= py*py;

    psp0 += de*de;
    psp0 += (2.00/v0byc_d)*de;

    pz = sqrt(psp0);
    
    


    KLx = k1l*y + 2.00*k2l*x*y + kls0;
    KLy  = h*length + k1l*x - k1l*y*y/2.0*h + k2l*(x*x - y*y) + k0l;  //VR added kls0 and k0l for kicker field effects.
    
    vKL = (px*KLx + py*KLy)/(p/p0_d);

    fx = (1.00 + GG_d*gamma)*KLx - GG_d*(gamma - 1.00)*vKL*px/(p/p0_d);
    fy = (1.00 + GG_d*gamma)*KLy - GG_d*(gamma - 1.00)*vKL*py/(p/p0_d);
    fz = -GG_d*(gamma - 1.00)*vKL*pz/(p/p0_d);

    dt_by_ds = (1.00 + h*x)/pz;

    fx *= dt_by_ds;
    fy *= dt_by_ds;
    fz *= dt_by_ds;

    omega = sqrt(fx*fx + (fy - h*length)*(fy - h*length) + fz*fz);
    
     if( omega > small ) {
    
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

    sx1 = s_mat00*sx0 + s_mat01*sy0 + s_mat02*sz0;
    sy1 = s_mat10*sx0 + s_mat11*sy0 + s_mat12*sz0;
    sz1 = s_mat20*sx0 + s_mat21*sy0 + s_mat22*sz0;
    sx0=sx1;sy0=sy1;sz0=sz1;
   }

__device__ void gpu3dmatrixP(precision s_mat0, precision s_mat1,precision s_mat2, precision s_mat3, precision s_mat4, precision s_mat5, precision s_mat6, precision s_mat7, precision s_mat8, int N, volatile precision &sx, volatile precision &sy, volatile precision &sz)
{
 precision SX_out, SY_out, SZ_out;
   
    SX_out = sx*s_mat0 + sy*s_mat1 + sz*s_mat2;
    SY_out = sx*s_mat3 + sy*s_mat4 + sz*s_mat5;
    SZ_out = sx*s_mat6 + sy*s_mat7 + sz*s_mat8;

    sx = SX_out; sy = SY_out; sz = SZ_out;
return; 

}
__device__ void SnakeProp(int N,int j, volatile precision &sx, volatile precision &sy, volatile precision &sz)
{
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


     
  
  gpu3dmatrixP(s_mat[0],s_mat[1],s_mat[2],s_mat[3],s_mat[4],s_mat[5],s_mat[6],s_mat[7],s_mat[8],N,sx,sy,sz);
 
}


__device__ void propagateSpin(int N, int j, volatile precision x, volatile precision px, volatile precision y, volatile precision py, volatile precision ct, volatile precision de, volatile precision &sx, volatile precision &sy, volatile precision &sz)
{
  //printf("propagateSpin \n"); 
  int ns = 1;
  if(rhic_d[j].ns > 0) ns = 4*rhic_d[j].ns;
  
  precision length;
  length = rhic_d[j].length/ns;

  precision ang,h; 
  ang = 0.00; h = 0.00;
  if(rhic_d[j].bend != 0)  {
      ang    = rhic_d[j].bend/ns;
// VR added to avoid div by zero Dec22 2010
     if(length > small)
      h      = ang/length;
  }

  precision k1l,k2l,k0l,kls0;
  k1l = 0.00; k2l = 0.00;
  k0l = 0.00; kls0 = 0.00; // VR added to handle hkicker and vkicker spin effects Dec22 2010

  /**
  if(rhic_d[j].mlt[0] != 1000){
    if(rhic_d[j].order == 0){
      k0l = rhic_d[j].k0l; kls0 = rhic_d[j].kls0 ;} // VR added to handle hkicker and vkicker spin effects Dec22 2010
    if(rhic_d[j].order > 0) k1l   =   rhic_d[j].k1l;
    if(rhic_d[j].order > 1) k2l   =   rhic_d[j].k2l;
  }
  **/
  k1l = rhic_d[j].k1l; k0l= rhic_d[j].k0l; kls0 = rhic_d[j].kls0;
  k2l = rhic_d[j].k2l;
  //if(length != 0 || k1l != 0 || k2l != 0) {
   gpupropogateSpin(N, ang, h, length, k1l, k2l, k0l,kls0, x,px,y,py,ct,de, sx, sy, sz);
    

   //} 
    

}

__global__ void gpuPropagate(int N, int Nturns){
 
   int i = blockDim.x*blockIdx.x + threadIdx.x;
   precision length,ang;
   precision v;
   volatile precision x,px,y,py,ct,de,sx,sy,sz;
   x= pos_d[i].x; px = pos_d[i].px; y = pos_d[i].y;
   py = pos_d[i].py; ct = pos_d[i].ct; de = pos_d[i].de;
   sx = pos_d[i].sx; sy = pos_d[i].sy ; sz = pos_d[i].sz;
   bool MULT = false;
   for(int turns= 1; turns <= Nturns; turns++) {
 
 // looping over lattice elements 
for(int j = 0 ; j < 1845 ; j++)
   {
     length=0.00; ang = 0.00; MULT = false;
     v = p0_d/gam_d/m0_d/clite_d;
     if(rhic_d[j].mlt[0] < 1000) MULT = true;
 
       ang = fabs(rhic_d[j].bend); 
       length = rhic_d[j].length;
     if(rhic_d[j].ns == 0) {
    
       length   /= 2.00;
      
       if(rhic_d[j].rfcav == 1 ){
	
	 RFProp(N,j, x, px, y, py,ct,de);
	 continue;
       }
          

	 if(ang > small){
	  
	   BendProp(N,j,x,px,y,py,ct,de);
	 
	 }else if(MULT) {
	   MultProp(N,j,x,px,y,py,ct,de);
	 }else if(length > 0) {
	   DriftProp(N,j,x,px,y,py,ct,de);
	 }

  
        t0_d += length/v;

	if(rhic_d[j].snake > 0 ){
  
	  SnakeProp(N,j,sx,sy,sz);} else{
	  propagateSpin(N,j,x,px,y,py,ct,de,sx,sy,sz);}



    if(ang > small){
	
      BendProp(N,j,x,px,y,py,ct,de);
	
    }else if(MULT) {
	
      MultProp(N,j,x,px,y,py,ct,de);
	
 } else if( length > small){
   
      DriftProp(N,j,x,px,y,py,ct,de);
      
}
   
    

    t0_d += length/v;
    
    continue;
     }

    
 
   
     int ns = rhic_d[j].ns*4;

   length /= 2*ns;
      


  for(int ii=0; ii < ns; ii++) {

    
   if(ang > small){
	
     BendProp(N,j,x,px,y,py,ct,de);

 }else if(MULT) {

     MultProp(N,j,x,px,y,py,ct,de);
	
 }  else if( length > small){
     
     DriftProp(N,j,x,px,y,py,ct,de);
      
   }
   
 
    t0_d += length/v;
    
    if(rhic_d[j].snake > 0){
     
      SnakeProp(N,j,sx,sy,sz);} else{
      propagateSpin(N,j,x,px,y,py,ct,de,sx,sy,sz);}
          

    if(ang > small){
	
      BendProp(N,j,x,px,y,py,ct,de);

    }else if(MULT) {

      MultProp(N,j,x,px,y,py,ct,de);
	  
    } else if( length > small){
     
      DriftProp(N,j,x,px,y,py,ct,de);
      
   }
   
             
 
    t0_d += length/v;
    
  }
   

   }

   }

   pos_d[i].x = x; pos_d[i].px = px; pos_d[i].y = y ; pos_d[i].py = py;
   pos_d[i].ct = ct; pos_d[i].de = de; pos_d[i].sx = sx; pos_d[i].sy = sy;
   pos_d[i].sz = sz;

}











#endif
     
   
