include $(UAL_ETEAPOT)/src/Makefile.config

INC  += -I$(UAL_PAC)/src
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPac

DIRS = ./Base  ./Integrator

OBJS = 

include ./Base/Objects
include ./Integrator/Objects

compile : 
	if [ ! -d $(UAL_ETEAPOT)/include/ETEAPOT ] ; \
		then (mkdir $(UAL_ETEAPOT)/include/ETEAPOT;); fi;
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done

clean:
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 	


