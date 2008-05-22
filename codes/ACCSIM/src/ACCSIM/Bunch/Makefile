include $(UAL_ACCSIM)/src/Makefile.config

LIBS += -lPacOptics -lPac

HHS = 	BunchAnalyzer.hh \
	BunchGenerator.hh 

include ./Objects

compile : $(OBJS)
	(if [ !  -d $(UAL_ACCSIM)/include/ACCSIM/Bunch ]; \
		then mkdir $(UAL_ACCSIM)/include/ACCSIM/Bunch; fi;)
	cp $(HHS) $(UAL_ACCSIM)/include/ACCSIM/Bunch/.	

clean:
	rm -rf $(OBJS) 
	rm -rf ../../../include/ACCSIM/Bunch


