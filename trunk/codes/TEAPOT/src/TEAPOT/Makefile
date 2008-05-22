include $(UAL_TEAPOT)/src/Makefile.config

INC  += -I$(UAL_PAC)/src
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPac

DIRS = ./Base  ./Integrator ./StringSC

OBJS = 

include ./Base/Objects
include ./Integrator/Objects
include ./StringSC/Objects

compile : 
	if [ ! -d $(UAL_TEAPOT)/include/TEAPOT ] ; \
		then (mkdir $(UAL_TEAPOT)/include/TEAPOT;); fi;	
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done

clean:
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 	


