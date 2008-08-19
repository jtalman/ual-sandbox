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


#include "UAL/ADXF/Writer.hh"


using namespace UAL;

MODULE = UAL::ADXF::Parser		PACKAGE = UAL::ADXF::Parser		


ADXFWriter*
ADXFWriter::new()


void
ADXFWriter::write(outFile)
		char* outFile
