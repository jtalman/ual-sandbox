#include "PAC/Beam/Bunch.hh"

#ifdef __cplusplus
extern "C" {
#endif
#include "EXTERN.h"
#include "perl.h"
#include "XSUB.h"
#ifdef __cplusplus
}
#endif

using namespace PAC;


MODULE = Pac::Beam           PACKAGE = Pac::BeamAttributes

BeamAttributes*
BeamAttributes::new()

void 
BeamAttributes::DESTROY()

double
BeamAttributes::energy(...)
   CODE:	
   double value;
   if(items == 1) RETVAL = THIS->getEnergy(); 
   if(items == 2) {
        value = (double ) SvNV(ST(1));
        THIS->setEnergy(value);
        RETVAL = value;
   }  
   OUTPUT:
   RETVAL 

double 
BeamAttributes::charge(...)
   CODE:	
   double value;
   if(items == 1) RETVAL = THIS->getCharge(); 
   if(items == 2) {
        value = (double ) SvNV(ST(1));
        THIS->setCharge(value);
        RETVAL = value;
   }  
   OUTPUT:
   RETVAL 


double
BeamAttributes::mass(...)
   CODE:	
   double value;
   if(items == 1) RETVAL = THIS->getMass(); 
   if(items == 2) {
        value = (double ) SvNV(ST(1));
        THIS->setMass(value);
        RETVAL = value;
   }  
   OUTPUT:
   RETVAL 

double
BeamAttributes::revfreq(...)
   CODE:	
   double value;
   if(items == 1) RETVAL = THIS->getRevfreq(); 
   if(items == 2) {
        value = (double ) SvNV(ST(1));
        THIS->setRevfreq(value);
        RETVAL = value;
   }  
   OUTPUT:
   RETVAL 

double
BeamAttributes::macrosize(...)
   CODE:	
   double value;
   if(items == 1) RETVAL = THIS->getMacrosize(); 
   if(items == 2) {
        value = (double ) SvNV(ST(1));
        THIS->setMacrosize(value);
        RETVAL = value;
   }  
   OUTPUT:
   RETVAL 

MODULE = Pac::Beam           PACKAGE = Pac::Bunch

Bunch*
Bunch::new(size)
    int size

void
Bunch::DESTROY()

int 
Bunch::size()

void 
Bunch::add(bunch)
   Bunch* bunch
   CODE:
   (*THIS).add(*bunch);

Position*
Bunch::position(...)
   CODE:	
   char* CLASS = "Pac::Position";
   int index = (int ) SvIV(ST(1));
   if(items == 3) {
            if(SvTYPE(SvRV(ST(2))) == SVt_PVMG) 
	    {
                (*THIS)[index].setPosition(*((Position *) SvIV((SV*) SvRV( ST(2) ))) );
            } 
            else
	    {
                warn( "PacBunch::position(...) -- arguments are not a blessed SV reference" );
                XSRETURN_UNDEF;
            }
   }
   RETVAL = new Position((*THIS)[index].getPosition());
   OUTPUT:
   RETVAL 

int
Bunch::flag(...)
   CODE:	
   int value;
   int index = (int ) SvIV(ST(1));
   if(items == 2) { RETVAL =(*THIS)[index].getFlag(); }
   if(items == 3) {
        value = (int ) SvIV(ST(2));
 	(*THIS)[index].setFlag(value);
        RETVAL = value;
   }  
   OUTPUT:
   RETVAL 

double
Bunch::energy(...)
   CODE:	
   double value;
   if(items == 1) RETVAL = THIS->getBeamAttributes().getEnergy(); 
   if(items == 2) {
        value = (double ) SvNV(ST(1));
        THIS->getBeamAttributes().setEnergy(value);
        RETVAL = value;
   }  
   OUTPUT:
   RETVAL 

double 
Bunch::charge(...)
   CODE:	
   double value;
   if(items == 1) RETVAL = THIS->getBeamAttributes().getCharge(); 
   if(items == 2) {
        value = (double ) SvNV(ST(1));
        THIS->getBeamAttributes().setCharge(value);
        RETVAL = value;
   }  
   OUTPUT:
   RETVAL 


double
Bunch::mass(...)
   CODE:	
   double value;
   if(items == 1) RETVAL = THIS->getBeamAttributes().getMass(); 
   if(items == 2) {
        value = (double ) SvNV(ST(1));
        THIS->getBeamAttributes().setMass(value);
        RETVAL = value;
   }  
   OUTPUT:
   RETVAL 


double
Bunch::revfreq(...)
   CODE:	
   double value;
   if(items == 1) RETVAL = THIS->getBeamAttributes().getRevfreq(); 
   if(items == 2) {
        value = (double ) SvNV(ST(1));
        THIS->getBeamAttributes().setRevfreq(value);
        RETVAL = value;
   }  
   OUTPUT:
   RETVAL 

double
Bunch::macrosize(...)
   CODE:	
   double value;
   if(items == 1) RETVAL = THIS->getBeamAttributes().getMacrosize(); 
   if(items == 2) {
        value = (double ) SvNV(ST(1));
        THIS->getBeamAttributes().setMacrosize(value);
        RETVAL = value;
   }  
   OUTPUT:
   RETVAL 

MODULE = Pac::Beam		PACKAGE = Pac::Position

Position*
Position::new()

void 
Position::DESTROY()

void 
Position::set(x, px, y, py, ct, de)
	double x
	double px
	double y
	double py
	double ct
       	double de
	
int 
Position::size()

double
Position::x(...)
   CODE:
   double value;
   if(items == 1) RETVAL = THIS->getX(); 
   if(items == 2) {
        value = (double ) SvNV(ST(1));
        THIS->setX(value);
        RETVAL = value;
   }  
   OUTPUT:
   RETVAL 

double
Position::px(...)
   CODE:
   double value;
   if(items == 1) RETVAL = THIS->getPX(); 
   if(items == 2) {
        value = (double ) SvNV(ST(1));
        THIS->setPX(value);
        RETVAL = value;
   }  
   OUTPUT:
   RETVAL 

double
Position::y(...)
   CODE:
   double value;
   if(items == 1) RETVAL = THIS->getY(); 
   if(items == 2) {
        value = (double ) SvNV(ST(1));
        THIS->setY(value);
        RETVAL = value;
   }  
   OUTPUT:
   RETVAL 

double
Position::py(...)
   CODE:
   double value;
   if(items == 1) RETVAL = THIS->getPY(); 
   if(items == 2) {
        value = (double ) SvNV(ST(1));
        THIS->setPY(value);
        RETVAL = value;
   }  
   OUTPUT:
   RETVAL 

double
Position::ct(...)
   CODE:
   double value;
   if(items == 1) RETVAL = THIS->getCT(); 
   if(items == 2) {
        value = (double ) SvNV(ST(1));
        THIS->setCT(value);
        RETVAL = value;
   }  
   OUTPUT:
   RETVAL 

double
Position::de(...)
   CODE:
   double value;
   if(items == 1) RETVAL = THIS->getDE(); 
   if(items == 2) {
        value = (double ) SvNV(ST(1));
        THIS->setDE(value);
        RETVAL = value;
   }  
   OUTPUT:
   RETVAL 

Position*
Position::add(...)
	CODE:
	char* CLASS = "Pac::Position";
	if(SvTYPE(SvRV(ST(1))) == SVt_PVMG) 
	{
	        RETVAL = new Position(*THIS);
		(*RETVAL) += *((Position *) SvIV((SV*) SvRV( ST(1) ))); 
	}
	else
	{
		warn( "PacPosition::add(...) -- argument is not Position" );
             	XSRETURN_UNDEF;
	}
	OUTPUT:
	RETVAL

Position*
Position::multiply(...)
	CODE:	
	char* CLASS = "Pac::Position";
	if(SvNOK(ST(1)) != 0 || SvIOK(ST(1)) != 0)
	{
	        RETVAL = new Position(*THIS);		    
		(*RETVAL) *= (double ) SvNV( ST(1) );
	}
	else
	{
		warn( "PacPosition::multiply(...) -- argument is not double" );
             	XSRETURN_UNDEF;
	}
	OUTPUT:
	RETVAL
