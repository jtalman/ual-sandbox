include $(UAL_ACCSIM)/src/Makefile.config

INC += -I$(UAL_PAC)/src
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPac

DIRS = ./Base  ./Bunch  ./Collimator

OBJS = 

include ./Base/Objects
include ./Bunch/Objects
include ./Collimator/Objects

compile : 
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done
	$(DLD) $(DLDFLAGS) -o $(UAL_ACCSIM)/lib/$(UAL_ARCH)/libAccsim.so $(OBJS) $(LIBS)

clean:
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 	
	rm -f $(UAL_ACCSIM)/lib/$(UAL_ARCH)/libAccsim.so


