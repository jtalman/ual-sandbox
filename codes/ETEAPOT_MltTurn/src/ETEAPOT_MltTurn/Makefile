include $(UAL_ETEAPOT_MltTurn)/src/Makefile.config

INC  += -I$(UAL_PAC)/src
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPac

DIRS = ./Integrator

OBJS = 

include ./Integrator/Objects

compile : 
	if [ ! -d $(UAL_ETEAPOT_MltTurn)/include/ETEAPOT_MltTurn ] ; \
		then (mkdir $(UAL_ETEAPOT_MltTurn)/include/ETEAPOT_MltTurn;); fi;
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done

clean:
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 	
