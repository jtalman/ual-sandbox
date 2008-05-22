include $(UAL_SIMBAD)/src/Makefile.config

LIBS += -lPacOptics -lPacBeam

HHS = 	Def.hh \
	BasicPropagator.hh 

include ./Objects

compile : $(OBJS)
	(if [ !  -d $(UAL_SIMBAD)/include/SIMBAD/Base ]; \
		then mkdir $(UAL_SIMBAD)/include/SIMBAD/Base; fi;)
	cp $(HHS) $(UAL_SIMBAD)/include/SIMBAD/Base/.	

clean:
	rm -f $(OBJS)
	rm -rf ../../../include/SIMBAD/Base

