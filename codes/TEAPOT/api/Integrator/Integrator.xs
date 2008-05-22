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

#include "Integrator/TeapotIntegrator.h"
#include "Integrator/TeapotDAIntegrator.h"

using namespace ZLIB;
using namespace PAC;

MODULE = Teapot::Integrator		PACKAGE = Teapot::Element

TeapotElement*
TeapotElement::new(e)
    PacGenElement* e
    CODE:
    RETVAL = new TeapotElement(*e);
    OUTPUT:
    RETVAL
	
void 
TeapotElement::DESTROY()

void 
TeapotElement::propagate(survey)
	PacSurveyData* survey
	CODE:
	THIS->propagate(*survey);

MODULE = Teapot::Integrator		PACKAGE = Teapot::Integrator

TeapotIntegrator*
TeapotIntegrator::new()

void
TeapotIntegrator::DESTROY()

void
TeapotIntegrator::propagate(ge, att, p)
	PacGenElement* ge
	BeamAttributes* att
	Position* p
	CODE:
	THIS->propagate(*ge, *att, *p);


MODULE = Teapot::Integrator		PACKAGE = Teapot::DaIntegrator

TeapotDAIntegrator*
TeapotDAIntegrator::new()

void
TeapotDAIntegrator::DESTROY()

void
TeapotDAIntegrator::propagate(ge, att, p)
	PacGenElement* ge
	BeamAttributes* att
	VTps* p
	CODE:
	THIS->propagate(*ge, *att, *p);

