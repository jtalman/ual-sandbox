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


#include "UAL/SXF/Parser.hh"


using namespace UAL;

MODULE = UAL::SXF::Parser		PACKAGE = UAL::SXF::Parser		


SXFParser*
SXFParser::new()

void
SXFParser::read(inFile, outFile)
		char* inFile
		char* outFile
	
void
SXFParser::write(outFile)
		char* outFile
