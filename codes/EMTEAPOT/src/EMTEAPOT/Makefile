include $(UAL_EMTEAPOT)/src/Makefile.config

INC  += -I$(UAL_PAC)/src
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPac

DIRS = ./Integrator

OBJS = 

include ./Integrator/Objects

compile : 
	if [ ! -d $(UAL_EMTEAPOT)/include/EMTEAPOT ] ; \
		then (mkdir $(UAL_EMTEAPOT)/include/EMTEAPOT;); fi;
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done

clean:
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 	
