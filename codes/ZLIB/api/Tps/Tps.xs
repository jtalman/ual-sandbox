#include "ZLIB/Tps/VTps.hh"


#ifdef __cplusplus
extern "C" {
#endif
#include "EXTERN.h"
#include "perl.h"
#include "XSUB.h"
#ifdef __cplusplus
}
#endif

using namespace ZLIB;

MODULE = Zlib::Tps		PACKAGE = Zlib::Space

Space*
Space::new(dim, order)
	int dim
	int order
	CODE:
	RETVAL = new Space(dim, order);
	OUTPUT:
	RETVAL

int
Space::mltOrder(...)
	CODE:
	if(items == 1) RETVAL = THIS->mltOrder();
	if(items == 2) {
	  int value = (int ) SvIV(ST(1));
          THIS->mltOrder(value);
	  RETVAL = value;
	}
	OUTPUT:
	RETVAL
	

void
Space::DESTROY()

MODULE = Zlib::Tps		PACKAGE = Zlib::Tps

Tps*
Tps::new()

void
Tps::DESTROY()

int
Tps::size()

int
Tps::order(...)
	CODE:
	if(items == 1) RETVAL = THIS->order(); 
        if(items == 2) {
	     int o = (int ) SvIV(ST(1));
             THIS->order(o);
             RETVAL = o;
	}
        OUTPUT:
        RETVAL

double
Tps::value(...)
        CODE:

	int index;

	if(SvIOK(ST(1))){
	   index = (int ) SvIV( ST(1) );
	}
        else{
	  if(SvNOK(ST(1))){ index = (int ) SvNV( ST(1) );}
	  else{
	     warn( "ZlibTps::value(...) -- index is not defined" );
             XSRETURN_UNDEF;	  
	  }
        }	  
	if(items == 2) RETVAL = (*THIS)[index]; 
        if(items == 3) {
	     double v = (double ) SvNV(ST(2));
             (*THIS)[index] = v;
             RETVAL = v;
	}
        OUTPUT:
        RETVAL

Tps* 
Tps::add(...)
	CODE:
        char* CLASS = "Zlib::Tps";
	RETVAL  = new Tps(*THIS);

        int flag = 0;

	if(SvNOK(ST(1))){
	  *RETVAL += (double ) SvNV( ST(1) );
           flag += 1;
	}
	if(SvIOK(ST(1))){
	    *RETVAL += (double ) SvIV( ST(1) );
            flag += 1;
	}	  
        if(SvROK(ST(1))){
	  if(SvTYPE(SvRV(ST(1))) == SVt_PVMG ){
             *RETVAL += *((Tps *) SvIV((SV*) SvRV( ST(1) ))); 
             flag += 1;
	  }
        }
        if(!flag)
	{
	    warn( "ZLIB::Tps::add(...) -- argument is not double or a blessed SV reference" );
            XSRETURN_UNDEF;
	}
	OUTPUT:
	RETVAL

Tps* 
Tps::multiply(...)
	CODE:
        char* CLASS = "Zlib::Tps";
	RETVAL  = new Tps(*THIS);

        int flag = 0;

	if(SvNOK(ST(1))){
	  *RETVAL *= (double ) SvNV( ST(1) );
           flag += 1;
	}
	if(SvIOK(ST(1))){
	  *RETVAL *= (double ) SvIV( ST(1) );
           flag += 1;
	}	  
        if(SvROK(ST(1))){
	  if(SvTYPE(SvRV(ST(1))) == SVt_PVMG ){
             *RETVAL *= *((Tps *) SvIV((SV*) SvRV( ST(1) ))); 
             flag += 1;
	  }
        }
        if(!flag)
	{
	    warn( "ZLIB::Tps::multiply(...) -- argument is not double or a blessed SV reference" );
            XSRETURN_UNDEF;
	}
	OUTPUT:
	RETVAL

Tps* 
Tps::subtract(...)
	CODE:
 	char* CLASS = "Zlib::Tps";
	RETVAL  = new Tps(*THIS);	    

        int flag = 0;

	if(SvNOK(ST(1))){
	  *RETVAL -= (double ) SvNV( ST(1) );
           flag += 1;
	}
	if(SvIOK(ST(1))){
	  *RETVAL -= (double ) SvIV( ST(1) );
           flag += 1;
	}	  
 	if(SvROK(ST(1))){       
	  if(SvTYPE(SvRV(ST(1))) == SVt_PVMG ){
             *RETVAL -= *((Tps *) SvIV((SV*) SvRV( ST(1) ))); 
             flag += 1;
	  }
        }
        if(!flag)
	{
	    warn( "ZLIB::Tps::subtract(...) -- argument is not double or a blessed SV reference" );
            XSRETURN_UNDEF;
	}

        int r = (int) SvIV( ST(2) );
        if(r) *RETVAL *= -1.;

	OUTPUT:
	RETVAL

Tps* 
Tps::divide(...)
	CODE:
        char* CLASS = "Zlib::Tps";

        int flag = 0;

        int r = (int) SvIV( ST(2) );

	if(SvNOK(ST(1))){
           if(r){
     	     RETVAL  = new Tps((double ) SvNV( ST(1) ), 0);
	    *RETVAL /= (*THIS);
           }
           else{
	      RETVAL  = new Tps(*THIS);
	     *RETVAL /= (double ) SvNV( ST(1) );
	   }
           flag += 1;
	}	
	if(SvIOK(ST(1))){
           if(r){
     	      RETVAL  = new Tps((double ) SvIV( ST(1) ), 0);
	     *RETVAL /= (*THIS);
           }
           else{
	      RETVAL  = new Tps(*THIS);
	     *RETVAL /= (double ) SvIV( ST(1) );
	   }
           flag += 1;
	}
	if(SvROK(ST(1))){  
	   if(SvTYPE(SvRV(ST(1))) == SVt_PVMG ){
             if(r){
               RETVAL   = new Tps(*((Tps *) SvIV((SV*) SvRV( ST(1) ))));
	      *RETVAL  /= (*THIS);
             }
             else{
	       RETVAL  = new Tps(*THIS);
              *RETVAL /= *((Tps *) SvIV((SV*) SvRV( ST(1) ))); 
	     }
             flag += 1;
           }
        }
        if(!flag)
	{
	    warn( "ZLIB::Tps::divide(...) -- argument is not double or a blessed SV reference" );
            XSRETURN_UNDEF;
	}

	OUTPUT:
	RETVAL

Tps*
Tps::D(iv)
       int iv
	CODE:
	char* CLASS = "Zlib::Tps";

	RETVAL = new Tps;
       *RETVAL = D(*THIS, iv);

        OUTPUT:
	RETVAL

Tps*
Tps::poisson(tps)
	Tps* tps
	CODE:
	char* CLASS = "Zlib::Tps";
	
        RETVAL = new Tps;
       *RETVAL = poisson(*THIS, *tps);

	OUTPUT:
	RETVAL

VTps*
Tps::vpoisson(vtps)
	VTps* vtps
	CODE:
	char* CLASS = "Zlib::VTps";
	
        RETVAL = new VTps(vtps->size());
       *RETVAL = poisson(*THIS, *vtps);

	OUTPUT:
	RETVAL

Tps*
Tps::sqrt(...)
	CODE:
        char* CLASS = "Zlib::Tps";

	RETVAL  = new Tps;
       *RETVAL  = sqrt(*THIS);

	OUTPUT:
	RETVAL

MODULE = Zlib::Tps		PACKAGE = Zlib::VTps


VTps*
VTps::new(size)
	int size

void
VTps::DESTROY()

int
VTps::size()

int
VTps::order(...)
	CODE:
	if(items == 1) RETVAL = THIS->order(); 
        if(items == 2) {
	     int o = (int ) SvIV(ST(1));
             THIS->order(o);
             RETVAL = o;
	}
        OUTPUT:
        RETVAL

Tps*
VTps::value(...)
        CODE:  
	int index;
	char* CLASS = "Zlib::Tps"; 	

	if(SvIOK(ST(1))){
	   index = (int ) SvIV( ST(1) );
	}
        else{
	  if(SvNOK(ST(1))){ index = (int ) SvNV( ST(1) );}
	  else{
	     warn( "ZLIB::Tps::value(...) -- index is not defined" );
             XSRETURN_UNDEF;	  
	  }
        }	  
	if(items == 2) RETVAL = new Tps((*THIS)[index]); 
        if(items == 3) {
	     if(SvTYPE(SvRV(ST(2))) == SVt_PVMG ){
                 RETVAL = new Tps(*( (Tps *) SvIV((SV*) SvRV( ST(2) )) )); 
                (*THIS)[index] = *RETVAL;
	     }
             else{
	         warn( "ZLIB::VTps::value(...) -- argument is not blessed SV reference" );
                 XSRETURN_UNDEF;
	     }
        }
        OUTPUT:
        RETVAL

VTps* 
VTps::add(...)
	CODE:
        char* CLASS = "Zlib::VTps";
	RETVAL  = new VTps(*THIS);

        int flag = 0;

	if(SvNOK(ST(1))){
	  *RETVAL += (double ) SvNV( ST(1) );
           flag += 1;
	}
	if(SvIOK(ST(1))){
	    *RETVAL += (double ) SvIV( ST(1) );
            flag += 1;
	}	  
        if(SvROK(ST(1))){
	  if(SvTYPE(SvRV(ST(1))) == SVt_PVMG ){
             *RETVAL += *((VTps *) SvIV((SV*) SvRV( ST(1) ))); 
             flag += 1;
	  }
        }
        if(!flag)
	{
	    warn( "ZLIB::VTps::add(...) -- argument is not double or a blessed SV reference" );
            XSRETURN_UNDEF;
	}
	OUTPUT:
	RETVAL



VTps* 
VTps::subtract(...)
	CODE:
 	char* CLASS = "Zlib::VTps";
	RETVAL  = new VTps(*THIS);	    

        int flag = 0;

	if(SvNOK(ST(1))){
	  *RETVAL -= (double ) SvNV( ST(1) );
           flag += 1;
	}
	if(SvIOK(ST(1))){
	  *RETVAL -= (double ) SvIV( ST(1) );
           flag += 1;
	}	  
 	if(SvROK(ST(1))){       
	  if(SvTYPE(SvRV(ST(1))) == SVt_PVMG ){
             *RETVAL -= *((VTps *) SvIV((SV*) SvRV( ST(1) ))); 
             flag += 1;
	  }
        }
        if(!flag)
	{
	    warn( "ZLIB::VTps::subtract(...) -- argument is not double or a blessed SV reference" );
            XSRETURN_UNDEF;
	}

        int r = (int) SvIV( ST(2) );
        if(r) *RETVAL *= -1.;

	OUTPUT:
	RETVAL

VTps* 
VTps::multiply(...)
	CODE:
        char* CLASS = "Zlib::VTps";
        double v;

        int flag = 0;

        int r = (int) SvIV( ST(2) );

	if(SvNOK(ST(1))){
	   RETVAL  = new VTps(*THIS);
	  *RETVAL *= (double ) SvNV( ST(1) );
           flag += 1;
	}	
	if(SvIOK(ST(1))){
	   RETVAL  = new VTps(*THIS);
	  *RETVAL *= (double ) SvIV( ST(1) );
           flag += 1;
	}
	if(SvROK(ST(1))){  
	   if(SvTYPE(SvRV(ST(1))) == SVt_PVMG ){
             if(r){
               RETVAL   = new VTps(*((VTps *) SvIV((SV*) SvRV( ST(1) ))));
	      *RETVAL  *= (*THIS);
             }
             else{
	       RETVAL  = new VTps(*THIS);
              *RETVAL *= *((VTps *) SvIV((SV*) SvRV( ST(1) ))); 
	     }
             flag += 1;
           }
        }
        if(!flag)
	{
	    warn( "ZLIB::VTps::multiply(...) -- argument is not double or a blessed SV reference" );
            XSRETURN_UNDEF;
	}

	OUTPUT:
	RETVAL

VTps* 
VTps::divide(...)
	CODE:
        char* CLASS = "Zlib::VTps";
        double v;

        int flag = 0;

        int r = (int) SvIV( ST(2) );

	if(SvNOK(ST(1))){
           if(r){
     	     RETVAL  = new VTps(THIS->size());
            *RETVAL  = (double ) SvNV( ST(1) );
	    *RETVAL /= (*THIS);
           }
           else{
	      RETVAL  = new VTps(*THIS);
	     *RETVAL /= (double ) SvNV( ST(1) );
	   }
           flag += 1;
	}	
	if(SvIOK(ST(1))){
           if(r){
     	      RETVAL  = new VTps(THIS->size());
             *RETVAL  = (double ) SvNV( ST(1) );            
	     *RETVAL /= (*THIS);
           }
           else{
	      RETVAL  = new VTps(*THIS);
	     *RETVAL /= (double ) SvIV( ST(1) );
	   }
           flag += 1;
	}
	if(SvROK(ST(1))){  
	   if(SvTYPE(SvRV(ST(1))) == SVt_PVMG ){
             if(r){
               RETVAL   = new VTps(*((VTps *) SvIV((SV*) SvRV( ST(1) ))));
	      *RETVAL  /= (*THIS);
             }
             else{
	       RETVAL  = new VTps(*THIS);
              *RETVAL /= *((VTps *) SvIV((SV*) SvRV( ST(1) ))); 
	     }
             flag += 1;
           }
        }
        if(!flag)
	{
	    warn( "ZLIB::VTps::divide(...) -- argument is not double or a blessed SV reference" );
            XSRETURN_UNDEF;
	}

	OUTPUT:
	RETVAL

VTps*
VTps::D(iv)
       int iv
	CODE:
	char* CLASS = "Zlib::VTps";

	RETVAL = new VTps(THIS->size());
       *RETVAL = D(*THIS, iv);

        OUTPUT:
	RETVAL

void
VTps::read(file)
	char* file

void 
VTps::write(file)
	char* file