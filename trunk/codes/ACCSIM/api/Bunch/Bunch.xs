#include <assert.h> //shishlo for gcc compiler 12.15.00

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

#include "ACCSIM/Bunch/BunchGenerator.hh"
#include "ACCSIM/Bunch/BunchAnalyzer.hh"

using namespace PAC;
using namespace ACCSIM;

MODULE = Accsim::Bunch               PACKAGE = Accsim::BunchGenerator

BunchGenerator*
BunchGenerator::new()

void 
BunchGenerator::DESTROY()

void 
BunchGenerator::shift(bunch, kick)
	Bunch* bunch
	Position* kick
	CODE:
	THIS->shift(*bunch, *kick);

void
BunchGenerator::addUniformRectangles(bunch, halfWidth, seed)
	Bunch* bunch
	Position* halfWidth
	int seed 
	CODE:
	THIS->addUniformRectangles(*bunch, *halfWidth, seed);
	OUTPUT:
	seed

void
BunchGenerator::addGaussianRectangles(bunch, rms, cut, seed)
	Bunch* bunch
	Position* rms
        double cut
	int seed 
	CODE:
	THIS->addGaussianRectangles(*bunch, *rms, cut, seed);
	OUTPUT:
	seed

void 
BunchGenerator::addUniformEllipses(bunch, twiss, emittance, seed)
	Bunch* bunch
	PacTwissData* twiss
	Position* emittance
	int seed
	CODE:
	THIS->addUniformEllipses(*bunch, *twiss, *emittance, seed);
	OUTPUT:
	seed

void 
BunchGenerator::addBinomialEllipses(bunch, m, twiss, emittance, seed)
	Bunch* bunch
	double m
	PacTwissData* twiss
	Position* emittance
	int seed
	CODE:
	THIS->addBinomialEllipses(*bunch, m, *twiss, *emittance, seed);
	OUTPUT:
	seed

MODULE = Accsim::Bunch               PACKAGE = Accsim::BunchAnalyzer

BunchAnalyzer*
BunchAnalyzer::new()

void 
BunchAnalyzer::DESTROY()

void
BunchAnalyzer::getRMS(bunch, orbit, twiss, rms)
	Bunch* bunch
	Position* orbit
	PacTwissData* twiss
	Position* rms
	CODE:
	THIS->getRMS(*bunch, *orbit, *twiss, *rms);









