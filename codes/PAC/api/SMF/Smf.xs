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

#ifdef list
#undef list
#endif


#include <string>
#include "SMF/PacSmf.h"

using namespace ZLIB;

MODULE = Pac::Smf		PACKAGE = Pac::Smf 


PacSmf*
create()
        CODE:	
        char* CLASS = "Pac::Smf";
        RETVAL = new PacSmf();
	OUTPUT:
	RETVAL

void 
initialize(smf)
        PacSmf* smf
        CODE:

        string tmpStr;
        char   tmpChr1[80];
        char   tmpChr2[80];
        char   tmpChrN[80];

        int max_order = 10;
 
        PacElemKeyIterator eki;
        for(eki = smf->elemKeys()->begin(); eki != smf->elemKeys()->end(); eki++){
           tmpStr = (*eki).name();
           /* tmpStr.upcase(); */
           strncpy(tmpChr1, tmpStr.c_str(), tmpStr.length());
           tmpChr1[tmpStr.length()] = '\0';
           sprintf(tmpChr2, "main::%s", tmpChr1);
           sv_setref_pv(perl_get_sv(tmpChr2, TRUE), 
			"Pac::ElemKey", 
			(void*) new PacElemKey(*eki));
        }
	
	PacElemBucketKeyIterator bki;
	for(bki = smf->bucketKeys()->begin(); bki != smf->bucketKeys()->end(); bki++){
	   for(int aki = 0; aki < (*bki).size(); aki++){
	      tmpStr = (*bki)[aki].name();
              /* tmpStr.upcase(); */
	      strncpy(tmpChr1, tmpStr.c_str(), tmpStr.length()); 
              tmpChr1[tmpStr.length()] = '\0'; 
              sprintf(tmpChr2, "main::%s", tmpChr1);         
	      sv_setref_pv(perl_get_sv(tmpChr2, TRUE), 
			"Pac::ElemAttribKey", 
			(void*) new PacElemAttribKey((*bki)[aki]));
	      if((*bki).order())
                for(int order = 0; order < max_order; order++){
	           sprintf(tmpChrN, "main::%s%d", tmpChr1, order);
	           sv_setref_pv(perl_get_sv(tmpChrN, TRUE), 
				"Pac::ElemAttribKey", 
				(void*) new PacElemAttribKey((*bki)[aki](order)));  
                } 
           } 
        } 
      

void
PacSmf::DESTROY()

PacElemKeys* 
PacSmf::elemKeys()
	CODE:
	char* CLASS = "Pac::ElemKeys";
	RETVAL = THIS->elemKeys();
	OUTPUT:
	RETVAL	
	
PacElemBucketKeys* 
PacSmf::bucketKeys()
	CODE:
	char* CLASS = "Pac::ElemBucketKeys";
	RETVAL = THIS->bucketKeys();
	OUTPUT:
	RETVAL

PacGenElements* 
PacSmf::elements()
	CODE:
	char* CLASS = "Pac::GenElements";
	RETVAL = THIS->elements();
	OUTPUT:
	RETVAL

PacLines* 
PacSmf::lines()
	CODE:
	char* CLASS = "Pac::Lines";
	RETVAL =  THIS->lines();
	OUTPUT:
	RETVAL

PacLattices* 
PacSmf::lattices()
	CODE:
	char* CLASS = "Pac::Lattices";
	RETVAL = THIS->lattices();
	OUTPUT:
	RETVAL

MODULE = Pac::Smf            PACKAGE = Pac::ElemKey

void
PacElemKey::DESTROY()

int
PacElemKey::key()

char*
PacElemKey::name()
	CODE:
	int l = THIS->name().length();
        RETVAL = new char[l + 1];
	strncpy(RETVAL, THIS->name().c_str(), l);
	RETVAL[l] = '\0';
	OUTPUT:
	RETVAL

MODULE = Pac::Smf		PACKAGE = Pac::ElemKeys

	
int
PacElemKeys::size()

PacElemKeyIterator* 
PacElemKeys::begin()
	CODE:
	char* CLASS = "Pac::ElemKeyIterator";
	RETVAL = new PacElemKeyIterator(THIS->begin());
	OUTPUT:
	RETVAL

PacElemKeyIterator* 
PacElemKeys::end()
	CODE:
	char* CLASS = "Pac::ElemKeyIterator";
	RETVAL = new PacElemKeyIterator(THIS->end());
	OUTPUT:
	RETVAL

PacElemKeyIterator* 
PacElemKeys::find(key)
	int key
	CODE:
	char* CLASS = "Pac::ElemKeyIterator";
	RETVAL = new PacElemKeyIterator(THIS->find(key));
	OUTPUT:
	RETVAL


MODULE = Pac::Smf            PACKAGE = Pac::ElemKeyIterator

void 
PacElemKeyIterator::add(...)
	CODE: 
	++(*THIS);
	
int 
PacElemKeyIterator::ne(i, f)
	PacElemKeyIterator* i
	int f
	CODE:
	RETVAL = *THIS != *i;
	OUTPUT:
	RETVAL

int
PacElemKeyIterator::first()
	CODE:	
	RETVAL = (THIS->operator*()).key();
	OUTPUT:
	RETVAL

PacElemKey*
PacElemKeyIterator::second()
	CODE:	
	char* CLASS = "Pac::ElemKey";
	RETVAL = new PacElemKey(THIS->operator*());
	OUTPUT:
	RETVAL

MODULE = Pac::Smf            PACKAGE = Pac::ElemBucketKey

void
PacElemBucketKey::DESTROY()

int
PacElemBucketKey::key()

char*
PacElemBucketKey::name()
	CODE:
	int l = THIS->name().length();
        RETVAL = new char[l + 1];
	strncpy(RETVAL, THIS->name().c_str(), l);
	RETVAL[l] = '\0';
	OUTPUT:
	RETVAL

int
PacElemBucketKey::size()

int
PacElemBucketKey::order()

PacElemAttribKey*
PacElemBucketKey::attribKey(...)
	CODE:	
	char* CLASS = "Pac::ElemAttribKey";
	if(items == 2) RETVAL = new PacElemAttribKey(THIS->operator[]( (int) SvIV(ST(1))) );
        if(items == 3) RETVAL = new PacElemAttribKey(THIS->operator[]( (int) SvIV(ST(1)))( (int) SvIV(ST(2))) );
	OUTPUT:
	RETVAL

MODULE = Pac::Smf		PACKAGE = Pac::ElemBucketKeys
	
int
PacElemBucketKeys::size()

PacElemBucketKeyIterator* 
PacElemBucketKeys::begin()
	CODE:
	char* CLASS = "Pac::ElemBucketKeyIterator";
	RETVAL = new PacElemBucketKeyIterator(THIS->begin());
	OUTPUT:
	RETVAL

PacElemBucketKeyIterator* 
PacElemBucketKeys::end()
	CODE:
	char* CLASS = "Pac::ElemBucketKeyIterator";
	RETVAL = new PacElemBucketKeyIterator(THIS->end());
	OUTPUT:
	RETVAL

PacElemBucketKeyIterator* 
PacElemBucketKeys::find(key)
	int key
	CODE:
	char* CLASS = "Pac::ElemBucketKeyIterator";
	RETVAL = new PacElemBucketKeyIterator(THIS->find(key));
	OUTPUT:
	RETVAL


MODULE = Pac::Smf            PACKAGE = Pac::ElemBucketKeyIterator

void 
PacElemBucketKeyIterator::add(...)
	CODE: 
	++(*THIS);
	
int 
PacElemBucketKeyIterator::ne(i, f)
	PacElemBucketKeyIterator* i
	int f
	CODE:
	RETVAL = *THIS != *i;
	OUTPUT:
	RETVAL

int
PacElemBucketKeyIterator::first()
	CODE:	
	RETVAL = (THIS->operator*()).key();
	OUTPUT:
	RETVAL

PacElemBucketKey*
PacElemBucketKeyIterator::second()
	CODE:	
	char* CLASS = "Pac::ElemBucketKey";
	RETVAL = new PacElemBucketKey(THIS->operator*());
	OUTPUT:
	RETVAL

MODULE = Pac::Smf            PACKAGE = Pac::ElemAttribKey

void
PacElemAttribKey::DESTROY()

PacElemBucketKey*
PacElemAttribKey::bucketKey()
	CODE:	
	char* CLASS = "Pac::ElemBucketKey";
	RETVAL = new PacElemBucketKey(THIS->bucketKey());
	OUTPUT:
	RETVAL

char*
PacElemAttribKey::name()
	CODE:
	int l = THIS->name().length();
        RETVAL = new char[l + 1];
	strncpy(RETVAL, THIS->name().c_str(), l);
	RETVAL[l] = '\0';
	OUTPUT:
	RETVAL

int
PacElemAttribKey::index()

int
PacElemAttribKey::order()

PacElemBucket*
PacElemAttribKey::multiply(...)
	CODE:
	char* CLASS = "Pac::ElemBucket";
	RETVAL = new PacElemBucket((*THIS)*((double) SvNV(ST(1))));
	OUTPUT:
	RETVAL

MODULE = Pac::Smf            PACKAGE = Pac::ElemBucket

void
PacElemBucket::DESTROY()

int 
PacElemBucket::key()

int 
PacElemBucket::size()

double
PacElemBucket::value(index)
	int index
	CODE:
	RETVAL = THIS->operator[](index);
	OUTPUT:
	RETVAL


MODULE = Pac::Smf            PACKAGE = Pac::ElemAttributes

PacElemAttribIterator* 
PacElemAttributes::begin()
	CODE:
	char* CLASS = "Pac::ElemAttribIterator";
	RETVAL = new PacElemAttribIterator(THIS->begin());
	OUTPUT:
	RETVAL

PacElemAttribIterator* 
PacElemAttributes::end()
	CODE:
	char* CLASS = "Pac::ElemAttribIterator";
	RETVAL = new PacElemAttribIterator(THIS->end());
	OUTPUT:
	RETVAL

PacElemAttribIterator* 
PacElemAttributes::find(key)
	int key
	CODE:
	char* CLASS = "Pac::ElemAttribIterator";
	RETVAL = new PacElemAttribIterator(THIS->find(key));
	OUTPUT:
	RETVAL

void
PacElemAttributes::set(...)
	CODE:
	THIS->erase(THIS->begin(),THIS->end());
 	
	for(int i=1; i < items; i++){
            if(sv_isa(ST(i), "Pac::ElemBucket") && (SvTYPE(SvRV(ST(i))) == SVt_PVMG) )
	    {
                THIS->add( *((PacElemBucket *) SvIV((SV*) SvRV( ST(i) ))) );
            } 
	    else{ 
	       if(sv_isa(ST(i), "Pac::ElemAttributes") && (SvTYPE(SvRV(ST(i))) == SVt_PVMG) )
 	       {
                   THIS->add( *((PacElemAttributes *) SvIV((SV*) SvRV( ST(i) ))) );
               } 
               else
	       {
                  warn( "PacElemAttributes::set(...) -- arguments are not a blessed SV reference" );
                  XSRETURN_UNDEF;
               }
            }
        };


void
PacElemAttributes::add(...)
	CODE:
 	
	for(int i=1; i < items; i++){
            if(sv_isa(ST(i), "Pac::ElemBucket") && (SvTYPE(SvRV(ST(i))) == SVt_PVMG) )
	    {
                THIS->add( *((PacElemBucket *) SvIV((SV*) SvRV( ST(i) ))) );
            } 
	    else{ 
	       if(sv_isa(ST(i), "Pac::ElemAttributes") && (SvTYPE(SvRV(ST(i))) == SVt_PVMG) )
 	       {
                   THIS->add( *((PacElemAttributes *) SvIV((SV*) SvRV( ST(i) ))) );
               } 
               else
	       {
                  warn( "PacElemAttributes::add(...) -- arguments are not a blessed SV reference" );
                  XSRETURN_UNDEF;
               }
            }
        };

double
PacElemAttributes::get(key)
	PacElemAttribKey* key
	CODE:
	RETVAL = THIS->get(*key);
	OUTPUT:
	RETVAL

void
PacElemAttributes::remove(key)
	PacElemAttribKey* key
	CODE:
	THIS->remove(*key);

MODULE = Pac::Smf            PACKAGE = Pac::ElemAttribIterator

void 
PacElemAttribIterator::add(...)
	CODE: 
	++(*THIS);
	
int 
PacElemAttribIterator::ne(i, f)
	PacElemAttribIterator* i
	int f
	CODE:
	RETVAL = *THIS != *i;
	OUTPUT:
	RETVAL

int
PacElemAttribIterator::first()
	CODE:	
	RETVAL = (THIS->operator*()).key();
	OUTPUT:
	RETVAL

PacElemBucket*
PacElemAttribIterator::second()
	CODE:	
	char* CLASS = "Pac::ElemBucket";
	RETVAL = new PacElemBucket(THIS->operator*());
	OUTPUT:
	RETVAL

MODULE = Pac::Smf 		PACKAGE = Pac::ElemPart

void
PacElemPart::set(...)
	CODE:
	THIS->remove();
 	
	for(int i=1; i < items; i++){
            if(sv_isa(ST(i), "Pac::ElemBucket") && (SvTYPE(SvRV(ST(i))) == SVt_PVMG) )
	    {
                THIS->add( *((PacElemBucket *) SvIV((SV*) SvRV( ST(i) ))) );
            } 
	    else{ 
	       if(sv_isa(ST(i), "Pac::ElemAttributes") && (SvTYPE(SvRV(ST(i))) == SVt_PVMG) )
 	       {
                   THIS->add( *((PacElemAttributes *) SvIV((SV*) SvRV( ST(i) ))) );
               } 
               else
	       {
                  warn( "PacElemPart::set(...) -- arguments are not a blessed SV reference" );
                  XSRETURN_UNDEF;
               }
            }
        };


void
PacElemPart::add(...)
	CODE:
 	
	for(int i=1; i < items; i++){
            if(sv_isa(ST(i), "Pac::ElemBucket") && (SvTYPE(SvRV(ST(i))) == SVt_PVMG) )
	    {
                THIS->add( *((PacElemBucket *) SvIV((SV*) SvRV( ST(i) ))) );
            } 
	    else{ 
	       if(sv_isa(ST(i), "Pac::ElemAttributes") && (SvTYPE(SvRV(ST(i))) == SVt_PVMG) )
 	       {
                   THIS->add( *((PacElemAttributes *) SvIV((SV*) SvRV( ST(i) ))) );
               } 
               else
	       {
                  warn( "PacElemPart::add(...) -- arguments are not a blessed SV reference" );
                  XSRETURN_UNDEF;
               }
            }
        };

double
PacElemPart::get(key)
	PacElemAttribKey* key
	CODE:
	RETVAL = THIS->get(*key);
	OUTPUT:
	RETVAL

void
PacElemPart::remove(key)
	PacElemAttribKey* key
	CODE:
	THIS->remove(*key);

PacElemAttributes*
PacElemPart::attributes()
	CODE:
	char* CLASS = "Pac::ElemAttributes";
	RETVAL = &THIS->attributes();
	OUTPUT:
	RETVAL

PacElemAttributes*
PacElemPart::rms()
	CODE:
	char* CLASS = "Pac::ElemAttributes";
	RETVAL = &THIS->rms();
	OUTPUT:
	RETVAL

MODULE = Pac::Smf            PACKAGE = Pac::GenElement

PacGenElement*
PacGenElement::new(n, id)
	char* n
	int id

void 
PacGenElement::copy(right)
   PacGenElement* right
   CODE:
   PacElemPart* part;
   for(int i=0; i < 3; i++){
     part = right->getPart(i);
     if(part) { THIS->setPart(i)->set(part->attributes()); }
  }

char* 
PacGenElement::name()
	CODE:
	int l = THIS->name().length();
        RETVAL = new char[l + 1];
	strncpy(RETVAL, THIS->name().c_str(), l);
	RETVAL[l] = '\0';
	OUTPUT:
	RETVAL        

void
PacGenElement::DESTROY()

int
PacGenElement::key()

PacElemPart*
PacGenElement::body()
	CODE:
	char* CLASS = "Pac::ElemPart";
	RETVAL = &THIS->body();
	OUTPUT:
	RETVAL

PacElemPart*
PacGenElement::front()
	CODE:
	char* CLASS = "Pac::ElemPart";
	RETVAL = &THIS->front();
	OUTPUT:
	RETVAL

PacElemPart*
PacGenElement::end()
	CODE:
	char* CLASS = "Pac::ElemPart";
	RETVAL = &THIS->end();
	OUTPUT:
	RETVAL

PacElemPart*
PacGenElement::getPart(index)
	int index
	CODE:
	char* CLASS = "Pac::ElemPart";
	RETVAL = THIS->getPart(index);
	OUTPUT:
	RETVAL

PacElemAttributes*
PacGenElement::rms()
	CODE:
	char* CLASS = "Pac::ElemAttributes";
	RETVAL = &THIS->rms();
	OUTPUT:
	RETVAL

void
PacGenElement::set(...)
	CODE:
	THIS->remove();
 	
	for(int i=1; i < items; i++){
            if(sv_isa(ST(i), "Pac::ElemBucket") && (SvTYPE(SvRV(ST(i))) == SVt_PVMG) )
	    {
                THIS->add( *((PacElemBucket *) SvIV((SV*) SvRV( ST(i) ))) );
            } 
	    else{ 
	       if(sv_isa(ST(i), "Pac::ElemAttributes") && (SvTYPE(SvRV(ST(i))) == SVt_PVMG) )
 	       {
                   THIS->add( *((PacElemAttributes *) SvIV((SV*) SvRV( ST(i) ))) );
               } 
               else
	       {
                  warn( "PacGenElement::set(...) -- arguments are not a blessed SV reference" );
                  XSRETURN_UNDEF;
               }
            }
        };


void
PacGenElement::add(...)
	CODE:
 	
	for(int i=1; i < items; i++){
            if(sv_isa(ST(i), "Pac::ElemBucket") && (SvTYPE(SvRV(ST(i))) == SVt_PVMG) )
	    {
                THIS->add( *((PacElemBucket *) SvIV((SV*) SvRV( ST(i) ))) );
            } 
	    else{ 
	       if(sv_isa(ST(i), "Pac::ElemAttributes") && (SvTYPE(SvRV(ST(i))) == SVt_PVMG) )
 	       {
                   THIS->add( *((PacElemAttributes *) SvIV((SV*) SvRV( ST(i) ))) );
               } 
               else
	       {
                  warn( "PacGenElement::add(...) -- arguments are not a blessed SV reference" );
                  XSRETURN_UNDEF;
               }
            }
        };

double
PacGenElement::get(key)
	PacElemAttribKey* key
	CODE:
	RETVAL = THIS->get(*key);
	OUTPUT:
	RETVAL

void
PacGenElement::remove(key)
	PacElemAttribKey* key
	CODE:
	THIS->remove(*key);

void
PacGenElement::map(vtps)
	VTps* vtps
	CODE:
        PacVTps ptps(*vtps);
        THIS->map(ptps); 

MODULE = Pac::Smf		PACKAGE = Pac::GenElements


void 
PacGenElements::declare(...)
	CODE:

	PacElemKey* key;

        if(sv_isobject(ST(1)) && (SvTYPE(SvRV(ST(1))) == SVt_PVMG) )
                { key  = (PacElemKey *) SvIV((SV*) SvRV( ST(1) )); }
        else{
                warn( "PacGenElements::declare(...) -- first argument is not a blessed SV reference" );
                XSRETURN_UNDEF;
        };

	int id = key->key();
     
        char*  n; 
        int    l;
        char   tmpChr1[80];
        char   tmpChr2[80];
	for(int i = 2; i < items; i++){
	   n = (char *) SvPV(ST(i), PL_na); l =  strlen(n);
           strncpy(tmpChr1, n, l); tmpChr1[l] = '\0';
           sprintf(tmpChr2, "main::%s\0", tmpChr1 );
	   sv_setref_pv(perl_get_sv(tmpChr2, TRUE), "Pac::GenElement", (void*) new PacGenElement(n, id));
	}
	
int
PacGenElements::size()

PacGenElemIterator* 
PacGenElements::begin()
	CODE:
	char* CLASS = "Pac::GenElemIterator";
	RETVAL = new PacGenElemIterator(THIS->begin());
	OUTPUT:
	RETVAL

PacGenElemIterator* 
PacGenElements::end()
	CODE:
	char* CLASS = "Pac::GenElemIterator";
	RETVAL = new PacGenElemIterator(THIS->end());
	OUTPUT:
	RETVAL

PacGenElemIterator* 
PacGenElements::find(key)
	char* key
	CODE:
	char* CLASS = "Pac::GenElemIterator";
	RETVAL = new PacGenElemIterator(THIS->find(string(key)));
	OUTPUT:
	RETVAL

MODULE = Pac::Smf		PACKAGE = Pac::GenElemIterator


void 
PacGenElemIterator::add(...)
	CODE: 
	++(*THIS);
	
int 
PacGenElemIterator::ne(i, f)
	PacGenElemIterator* i
	int f
	CODE:
	RETVAL = *THIS != *i;
	OUTPUT:
	RETVAL

char*
PacGenElemIterator::first()
	CODE:
	int l = (THIS->operator*()).name().length();
        RETVAL = new char[l + 1];
	strncpy(RETVAL, (THIS->operator*()).name().c_str(), l);
	RETVAL[l] = '\0';	
	OUTPUT:
	RETVAL

PacGenElement*
PacGenElemIterator::second()
	CODE:	
	char* CLASS = "Pac::GenElement";
	RETVAL = new PacGenElement(THIS->operator*());
	OUTPUT:
	RETVAL

MODULE = Pac::Smf		PACKAGE = Pac::Line

PacLine*
PacLine::new(n)
	char* n

void
PacLine::DESTROY()

char* 
PacLine::name()
	CODE:
	int l = THIS->name().length();
        RETVAL = new char[l + 1];
	strncpy(RETVAL, THIS->name().c_str(), l);
	RETVAL[l] = '\0';
	OUTPUT:
	RETVAL  
    
void
PacLine::set(...)
	CODE:
	THIS->erase();
	for(int i=1; i < items; i++){
            if(sv_isa(ST(i), "Pac::GenElement") && (SvTYPE(SvRV(ST(i))) == SVt_PVMG) )
	    {
                THIS->add( *((PacGenElement *) SvIV((SV*) SvRV( ST(i) ))) );
            } 
	    else{ 
	       if(sv_isa(ST(i), "Pac::Line") && (SvTYPE(SvRV(ST(i))) == SVt_PVMG) )
 	       {
                   THIS->add( *((PacLine *) SvIV((SV*) SvRV( ST(i) ))) );
               } 
               else
	       {
                  warn( "PacLine::set(...) -- arguments are not a blessed SV reference" );
                  XSRETURN_UNDEF;
               }
            }
        };

void
PacLine::add(...)
	CODE:
	for(int i=1; i < items; i++){
            if(sv_isa(ST(i), "Pac::GenElement") && (SvTYPE(SvRV(ST(i))) == SVt_PVMG) )
	    {
                THIS->add( *((PacGenElement *) SvIV((SV*) SvRV( ST(i) ))) );
            } 
	    else{ 
	       if(sv_isa(ST(i), "Pac::Line") && (SvTYPE(SvRV(ST(i))) == SVt_PVMG) )
 	       {
                   THIS->add( *((PacLine *) SvIV((SV*) SvRV( ST(i) ))) );
               } 
               else
	       {
                  warn( "PacLine::add(...) -- arguments are not a blessed SV reference" );
                  XSRETURN_UNDEF;
               }
            }
        };

PacLine*
PacLine::multiply(...)
	CODE:	
	char* CLASS = "Pac::Line";
	RETVAL = new PacLine( ((int) SvIV(ST(1)))*(*THIS) );
	OUTPUT:
	RETVAL

MODULE = Pac::Smf            PACKAGE = Pac::Lines

void 
PacLines::declare(...)
	CODE:

	char* n;
	for(int i = 1; i < items; i++){
	   n = (char *) SvPV(ST(i), PL_na);
	   sv_setref_pv(perl_get_sv(n, TRUE), "Pac::Line", (void*) new PacLine(n));
	}
	
int
PacLines::size()

PacLineIterator* 
PacLines::begin()
	CODE:
	char* CLASS = "Pac::LineIterator";
	RETVAL = new PacLineIterator(THIS->begin());
	OUTPUT:
	RETVAL

PacLineIterator* 
PacLines::end()
	CODE:
	char* CLASS = "Pac::LineIterator";
	RETVAL = new PacLineIterator(THIS->end());
	OUTPUT:
	RETVAL

PacLineIterator* 
PacLines::find(key)
	char* key
	CODE:
	char* CLASS = "Pac::LineIterator";
	RETVAL = new PacLineIterator(THIS->find(string(key)));
	OUTPUT:
	RETVAL

MODULE = Pac::Smf		PACKAGE = Pac::LineIterator


void 
PacLineIterator::add(...)
	CODE: 
	++(*THIS);
	

int 
PacLineIterator::ne(i, f)
	PacLineIterator* i
	int f
	CODE:
	RETVAL = *THIS != *i;
	OUTPUT:
	RETVAL

char*
PacLineIterator::first()
	CODE:	
	int l = (THIS->operator*()).name().length();
        RETVAL = new char[l + 1];
	strncpy(RETVAL, (THIS->operator*()).name().c_str(), l);
	RETVAL[l] = '\0';
	OUTPUT:
	RETVAL

PacLine*
PacLineIterator::second()
	CODE:	
	char* CLASS = "Pac::Line";
	RETVAL = new PacLine(THIS->operator*());
	OUTPUT:
	RETVAL

MODULE = Pac::Smf		PACKAGE = Pac::LattElement

void
PacLattElement::DESTROY()

PacLattElement*
PacLattElement::new(genElement)
	PacGenElement* genElement
      	CODE:
        RETVAL = new PacLattElement(*genElement);
	OUTPUT:
	RETVAL

void
PacLattElement::setName(name)
	char* name
	CODE:	
	THIS->name(string(name));

char*
PacLattElement::name()
	CODE:	
	int l = THIS->name().length();
        RETVAL = new char[l + 1];
	strncpy(RETVAL, THIS->name().c_str(), l);
	RETVAL[l] = '\0';
	OUTPUT:
	RETVAL

char*
PacLattElement::type()
	CODE:	
	int l = THIS->type().length();
        RETVAL = new char[l + 1];
	strncpy(RETVAL, THIS->type().c_str(), l);
	RETVAL[l] = '\0';
	OUTPUT:
	RETVAL


char*
PacLattElement::genName()
	CODE:
	int l = THIS->genElement().name().length();
        RETVAL = new char[l + 1];
	strncpy(RETVAL, THIS->genElement().name().c_str(), l);
	RETVAL[l] = '\0';	
	OUTPUT:
	RETVAL

int
PacLattElement::key()
	CODE:	
	RETVAL = THIS->genElement().key();
	OUTPUT:
	RETVAL	

PacElemAttributes*
PacLattElement::body()
	CODE:
	char* CLASS = "Pac::ElemAttributes";
	RETVAL = &THIS->body();
	OUTPUT:
	RETVAL

PacElemAttributes*
PacLattElement::front()
	CODE:
	char* CLASS = "Pac::ElemAttributes";
	RETVAL = &THIS->front();
	OUTPUT:
	RETVAL

PacElemAttributes*
PacLattElement::end()
	CODE:
	char* CLASS = "Pac::ElemAttributes";
	RETVAL = &THIS->end();
	OUTPUT:
	RETVAL

PacElemAttributes*
PacLattElement::getPart(index)
	int index
	CODE:
	char* CLASS = "Pac::ElemAttributes";
	RETVAL = THIS->getPart(index);
	OUTPUT:
	RETVAL

void
PacLattElement::set(...)
	CODE:
	THIS->remove();
 	
	for(int i=1; i < items; i++){
            if(sv_isa(ST(i), "Pac::ElemBucket") && (SvTYPE(SvRV(ST(i))) == SVt_PVMG) )
	    {
                THIS->add( *((PacElemBucket *) SvIV((SV*) SvRV( ST(i) ))) );
            } 
	    else{ 
	       if(sv_isa(ST(i), "Pac::ElemAttributes") && (SvTYPE(SvRV(ST(i))) == SVt_PVMG) )
 	       {
                   THIS->add( *((PacElemAttributes *) SvIV((SV*) SvRV( ST(i) ))) );
               } 
               else
	       {
                  warn( "PacLattElement::set(...) -- arguments are not a blessed SV reference" );
                  XSRETURN_UNDEF;
               }
            }
        };


void
PacLattElement::add(...)
	CODE:
 	
	for(int i=1; i < items; i++){
            if(sv_isa(ST(i), "Pac::ElemBucket") && (SvTYPE(SvRV(ST(i))) == SVt_PVMG) )
	    {
                THIS->add( *((PacElemBucket *) SvIV((SV*) SvRV( ST(i) ))) );
            } 
	    else{ 
	       if(sv_isa(ST(i), "Pac::ElemAttributes") && (SvTYPE(SvRV(ST(i))) == SVt_PVMG) )
 	       {
                   THIS->add( *((PacElemAttributes *) SvIV((SV*) SvRV( ST(i) ))) );
               } 
               else
	       {
                  warn( "PacLattElement::add(...) -- arguments are not a blessed SV reference" );
                  XSRETURN_UNDEF;
               }
            }
        };

double
PacLattElement::get(key)
	PacElemAttribKey* key
	CODE:
	RETVAL = THIS->get(*key);
	OUTPUT:
	RETVAL

void
PacLattElement::remove(key)
	PacElemAttribKey* key
	CODE:
	THIS->remove(*key);

MODULE = Pac::Smf		PACKAGE = Pac::Lattice

PacLattice*
PacLattice::new(n)
	char* n

void
PacLattice::DESTROY()

char* 
PacLattice::name()
	CODE:
	int l = THIS->name().length();
        RETVAL = new char[l + 1];
	strncpy(RETVAL, THIS->name().c_str(), l);
	RETVAL[l] = '\0';
	OUTPUT:
	RETVAL    

void
PacLattice::set(...)
	CODE:
	for(int i=1; i < items; i++){
            if(sv_isa(ST(i), "Pac::Lattice") && (SvTYPE(SvRV(ST(i))) == SVt_PVMG) )
	    {
                THIS->add( *((PacLattice *) SvIV((SV*) SvRV( ST(i) ))) );
            } 
	    else{ 
	       if(i > 1) {
                  warn( "PacLattice::set(...) -- arguments are not  blessed Lattice references" );
                  XSRETURN_UNDEF;
               }		
	       if(sv_isa(ST(i), "Pac::Line") && (SvTYPE(SvRV(ST(i))) == SVt_PVMG) )
 	       {
                   THIS->set( *((PacLine *) SvIV((SV*) SvRV( ST(i) ))) );
               } 
               else
	       {
                  warn( "PacLattice::set(...) -- arguments are not a blessed Lattice reference" );
                  XSRETURN_UNDEF;
               }
            }
        };

void
PacLattice::add(...)
	CODE:
	for(int i=1; i < items; i++){
            if(sv_isa(ST(i), "Pac::Lattice") && (SvTYPE(SvRV(ST(i))) == SVt_PVMG) )
	    {
                THIS->add( *((PacLattice *) SvIV((SV*) SvRV( ST(i) ))) );
            } 
            else
	    {
                warn( "PacLattice::add(...) -- arguments are not a blessed Lattice reference" );
                XSRETURN_UNDEF;
	    }
        };

PacLattElement*
PacLattice::element(index)
	int index
	CODE:
	char* CLASS = "Pac::LattElement";
	RETVAL = new PacLattElement();
	*RETVAL = (*THIS)[index];
	OUTPUT:
	RETVAL

int
PacLattice::size()

MODULE = Pac::Smf            PACKAGE = Pac::Lattices

void 
PacLattices::declare(...)
	CODE:

	char* n;
	for(int i = 1; i < items; i++){
	   n = (char *) SvPV(ST(i), PL_na);
	   sv_setref_pv(perl_get_sv(n, TRUE), "Pac::Lattice", (void*) new PacLattice(n));
	}
	
int
PacLattice::size()

PacLatticeIterator* 
PacLattices::begin()
	CODE:
	char* CLASS = "Pac::LatticeIterator";
	RETVAL = new PacLatticeIterator(THIS->begin());
	OUTPUT:
	RETVAL

PacLatticeIterator* 
PacLattices::end()
	CODE:
	char* CLASS = "Pac::LatticeIterator";
	RETVAL = new PacLatticeIterator(THIS->end());
	OUTPUT:
	RETVAL

PacLatticeIterator* 
PacLattices::find(key)
	char* key
	CODE:
	char* CLASS = "Pac::LatticeIterator";
	RETVAL = new PacLatticeIterator(THIS->find(string(key)));
	OUTPUT:
	RETVAL

MODULE = Pac::Smf		PACKAGE = Pac::LatticeIterator

void 
PacLatticeIterator::add(...)
	CODE: 
	++(*THIS);
	

int 
PacLatticeIterator::ne(i, f)
	PacLatticeIterator* i
	int f
	CODE:
	RETVAL = *THIS != *i;
	OUTPUT:
	RETVAL

char*
PacLatticeIterator::first()
	CODE:	
	int l = (THIS->operator*()).name().length();
        RETVAL = new char[l + 1];
	strncpy(RETVAL, (THIS->operator*()).name().c_str(), l);
	RETVAL[l] = '\0';	
	OUTPUT:
	RETVAL

PacLattice*
PacLatticeIterator::second()
	CODE:	
	char* CLASS = "Pac::Lattice";
	RETVAL = new PacLattice(THIS->operator*());
	OUTPUT:
	RETVAL
