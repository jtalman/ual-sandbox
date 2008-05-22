include $(UAL_ACCSIM)/src/Makefile.config

INC += -I$(UAL_PAC)/src -I$(UAL_ZLIB)/src
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPac

HHS = 	FermiScatter.hh \
	BetheBlochSlower.hh \
	NuclearInteraction.hh \
	CollimatorTracker.hh \
	IMaterialPropagator.hh

include ./Objects

compile : $(OBJS)
	(if [ !  -d $(UAL_ACCSIM)/include/ACCSIM/Collimator ]; \
		then mkdir  $(UAL_ACCSIM)/include/ACCSIM/Collimator; fi;)
	cp  $(HHS) $(UAL_ACCSIM)/include/ACCSIM/Collimator/.

clean:
	rm -f $(OBJS)
	rm -rf ../include/ACCSIM/Collimator	
