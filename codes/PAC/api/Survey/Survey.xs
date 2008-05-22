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

#include "Survey/PacSurveySbend.h"
#include "Survey/PacSurveyDrift.h"

MODULE = Pac::Survey         PACKAGE = Pac::Survey

PacSurvey *
PacSurvey::new()

void
PacSurvey::DESTROY()

double
PacSurvey::x()

double
PacSurvey::y()

double
PacSurvey::z()

double
PacSurvey::suml()

double
PacSurvey::theta()

double
PacSurvey::phi()

double
PacSurvey::psi()


MODULE = Pac::Survey		PACKAGE = Pac::SurveyData

PacSurveyData *
PacSurveyData::new()

void
PacSurveyData::DESTROY()

double
PacSurveyData::x()
	CODE:
	RETVAL = THIS->survey().x();
	OUTPUT:
	RETVAL

double
PacSurveyData::y()
	CODE:
	RETVAL = THIS->survey().y();
	OUTPUT:
	RETVAL

double
PacSurveyData::z()
	CODE:
	RETVAL = THIS->survey().z();
	OUTPUT:
	RETVAL

double
PacSurveyData::suml()
	CODE:
	RETVAL = THIS->survey().suml();
	OUTPUT:
	RETVAL

double
PacSurveyData::theta()
	CODE:
	RETVAL = THIS->survey().theta();
	OUTPUT:
	RETVAL

double
PacSurveyData::phi()
	CODE:
	RETVAL = THIS->survey().phi();
	OUTPUT:
	RETVAL

double
PacSurveyData::psi()
	CODE:
	RETVAL = THIS->survey().psi();
	OUTPUT:
	RETVAL

MODULE = Pac::Survey		PACKAGE = Pac::SurveyDrift

PacSurveyDrift *
PacSurveyDrift::new(length)
	double length

void
PacSurveyDrift::DESTROY()

void
PacSurveyDrift::define(length)
	double length

void
PacSurveyDrift::propagate(sdata)
	PacSurveyData* sdata
	CODE:
	THIS->propagate(*sdata);

MODULE = Pac::Survey		PACKAGE = Pac::SurveySbend

PacSurveySbend *
PacSurveySbend::new(length, angle, rotation)
	double length
	double angle
	double rotation

void
PacSurveySbend::DESTROY()

void
PacSurveySbend::define(length, angle, rotation)
	double length
	double angle
	double rotation

void
PacSurveySbend::propagate(sdata)
	PacSurveyData* sdata
	CODE:
	THIS->propagate(*sdata);
