include $(UAL_ACCSIM)/src/Makefile.config

LIBS += -lPacOptics -lPacBeam

HHS = 	Def.hh \
	BasicPropagator.hh \
	GaussianGenerator.hh \
	RandomNumberGenerator.hh \
	UniformGenerator.hh

include ./Objects

compile : $(OBJS)
	(if [ !  -d $(UAL_ACCSIM)/include/ACCSIM/Base ]; \
		then mkdir $(UAL_ACCSIM)/include/ACCSIM/Base; fi;)
	cp $(HHS) $(UAL_ACCSIM)/include/ACCSIM/Base/.	

clean:
	rm -f $(OBJS)
	rm -rf ../../../include/ACCSIM/Base






