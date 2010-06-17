void getB1tw(PacElemMultipole* mult, const PacElemOffset* offset, const double rkicks, const double in1, const double in2, double& B1tw){
  double t0, x, y, dpx, dpy;

  double Bxtwd, Bytwd;

  x = in1; //  xdif
  y = in2; //  ydif

  dpx = 0.0;
  dpy = 0.0;

  if(offset) {
    x -= offset->dx();
    y -= offset->dy();
  }

  if(mult){

     int     index = mult->size();
     double* data = mult->data();

     double kl, ktl;    

//const Coordinates pos = p;
//#include "set_betalBefore.cc"
     if(index > 0){
          do {
                ktl = data[--index];
                kl  = data[--index];
                t0  = x*dpx;
                t0 -= y*dpy - kl;
                dpy  = x*dpy;
                dpy += y*dpx + ktl;    
                dpx  = t0;
          } while ( index > 0 ) ;
     }
    dpx *= rkicks;
     dpy *= rkicks;
  }

  dpx *= -1;

  Bxtwd = dpy;                              // JDT - Bxtwd = (L*Bx)/(p0/e)   10/19/2009
  Bytwd = -dpx;                             // JDT - Bytwd = (L*By)/(p0/e)

  B1tw = Bxtwd;

//p[1] += dpx;                              // px/p0
//p[3] += dpy;                              // py/p0

}
