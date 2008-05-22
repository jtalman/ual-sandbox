#include <assert.h> 

#ifdef __cplusplus
extern "C" {
#endif
#include "EXTERN.h"
#include "perl.h"
#include "XSUB.h"
#ifdef __cplusplus
}
#endif

#ifdef do_open
#undef do_open
#endif

#ifdef do_close
#undef do_close
#endif

#include "Optics/PacChromData.h"
#include "Optics/PacTMap.h"

using namespace ZLIB;
using namespace PAC;

MODULE = Pac::Optics		PACKAGE = Pac::TMap

PacTMap*
PacTMap::new(size)
	int size

void
PacTMap::DESTROY()

int
PacTMap::mltOrder(...)
	CODE:	
        if(items == 1) { RETVAL = THIS->mltOrder(); }
   	if(items == 2) {
            int order = (int) SvIV( ST(1) );
            THIS->mltOrder(order);
            RETVAL = order;
	}
        OUTPUT:
	RETVAL

Position*
PacTMap::refOrbit(...)
	CODE:
	char* CLASS = "Pac::Position";
        Position* p;		
        if(items == 1) RETVAL = new Position(THIS->refOrbit()); 
   	if(items == 2) {
	  if(SvTYPE(SvRV(ST(1))) == SVt_PVMG ){
        	p = (Position *) SvIV((SV*) SvRV( ST(1) ));
        	THIS->refOrbit(*p);
        	RETVAL = new Position(*p);
	  }
	  else{
	    warn( "PacTMap::refOrbit(...) -- argument is not double or a blessed SV reference" );
            XSRETURN_UNDEF;
	  }
	}
        OUTPUT:
	RETVAL

VTps*
PacTMap::daVTps(...)
	CODE:
	char* CLASS = "Zlib::VTps";
        ZLIB::VTps* p;		
        if(items == 1) RETVAL = new ZLIB::VTps(*(THIS->operator->())); 
   	if(items == 2) {
	  if(SvTYPE(SvRV(ST(1))) == SVt_PVMG ){
        	p = (ZLIB::VTps *) SvIV((SV*) SvRV( ST(1) ));
        	THIS->operator=(*p);
        	RETVAL = new ZLIB::VTps(*p);
	  }
	  else{
	    warn( "PacTMap::daVTps(...) -- argument is not double or a blessed SV reference" );
            XSRETURN_UNDEF;
	  }
	}
        OUTPUT:
	RETVAL

void
PacTMap::propagate(...)
	CODE:
        int turns = 1;
        if(items = 3){
		if(SvNOK(ST(2))){ turns = (int ) SvNV( ST(2) ); }
		if(SvIOK(ST(2))){ turns = (int ) SvIV( ST(2) ); }
	}	  
	if(SvTYPE(SvRV(ST(1))) == SVt_PVMG ){
        	THIS->propagate(*(Bunch *) SvIV((SV*) SvRV( ST(1) )), turns);
	}
	else{
	    warn( "PacTMap::propagate(...) -- argument is not double or a blessed SV reference" );
            XSRETURN_UNDEF;
	}

void 
PacTMap::read(nfile)
	char* nfile

void 
PacTMap::write(nfile)
	char* nfile


MODULE = Pac::Optics		PACKAGE = Pac::TwissData

PacTwissData *
PacTwissData::new()

void 
PacTwissData::DESTROY()

double
PacTwissData::dimension()

void
PacTwissData::copy(in)
	PacTwissData* in;
	CODE:
	THIS->operator=(*in);

double
PacTwissData::beta(...)
	CODE:
	int dim = (int) SvIV(ST(1));
	double value;
   	if(items == 2) RETVAL = THIS->beta(dim); 
   	if(items == 3) {
		value = (double ) SvNV(ST(2));
		THIS->beta(dim, value);
		RETVAL = value;
	}
	OUTPUT:
	RETVAL

double
PacTwissData::alpha(...)
	CODE:
	int dim = (int) SvIV(ST(1));
	double value;
   	if(items == 2) RETVAL = THIS->alpha(dim); 
   	if(items == 3) {
		value = (double ) SvNV(ST(2));
		THIS->alpha(dim, value);
		RETVAL = value;
	}
	OUTPUT:
	RETVAL

double
PacTwissData::mu(...)
	CODE:
	int dim = (int) SvIV(ST(1));
	double value;
   	if(items == 2) RETVAL = THIS->mu(dim); 
   	if(items == 3) {
		value = (double ) SvNV(ST(2));
		THIS->mu(dim, value);
		RETVAL = value;
	}
	OUTPUT:
	RETVAL


double
PacTwissData::d(...)
	CODE:
	int dim = (int) SvIV(ST(1));
	double value;
   	if(items == 2) RETVAL = THIS->d(dim); 
   	if(items == 3) {
		value = (double ) SvNV(ST(2));
		THIS->d(dim, value);
		RETVAL = value;
	}
	OUTPUT:
	RETVAL

double
PacTwissData::dp(...)
	CODE:
	int dim = (int) SvIV(ST(1));
	double value;
   	if(items == 2) RETVAL = THIS->dp(dim); 
   	if(items == 3) {
		value = (double ) SvNV(ST(2));
		THIS->dp(dim, value);
		RETVAL = value;
	}
	OUTPUT:
	RETVAL

MODULE = Pac::Optics		PACKAGE = Pac::ChromData

PacChromData *
PacChromData::new()

void 
PacChromData::DESTROY()

double
PacChromData::dimension()

void
PacChromData::copy(in)
	PacChromData* in;
	CODE:
	THIS->operator=(*in);

PacTwissData*
PacChromData::twiss()
   CODE:	
   char* CLASS = "Pac::TwissData";
   RETVAL = new PacTwissData((*THIS).twiss());	
   OUTPUT:
   RETVAL


double
PacChromData::dmu(dim)
	int dim
	CODE:
	RETVAL = THIS->dmu(dim);
	OUTPUT:
	RETVAL


