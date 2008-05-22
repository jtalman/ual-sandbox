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

#include "Math/TeapotRandomGenerator.h"

MODULE = Teapot::Math		PACKAGE = Teapot::RandomGenerator

TeapotRandomGenerator* 
TeapotRandomGenerator::new(iseed)
	double iseed
	CODE:
	RETVAL = new TeapotRandomGenerator((int) iseed);
	OUTPUT:
	RETVAL

void 
TeapotRandomGenerator::DESTROY()

double
TeapotRandomGenerator::getSeed()
	CODE:
	RETVAL = (double) THIS->getSeed();
	OUTPUT:
	RETVAL

void 
TeapotRandomGenerator::setSeed(iseed)
	double iseed
	CODE:
	THIS->setSeed((int) iseed);

double
TeapotRandomGenerator::getran(cut)
	double cut
	CODE:
	RETVAL = THIS->getran((int) cut);
	OUTPUT:
	RETVAL

