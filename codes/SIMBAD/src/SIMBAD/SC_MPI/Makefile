include $(UAL_SIMBAD)/src/Makefile.config

INC += -I$(UAL_PAC)/include -I$(UAL)/tools/include

ifdef MPIHOME
INC += -I$(MPIHOME)/include
endif

include ./Objects

HHS = 	TSCPropagatorFFT_MPI.hh \
	TSCCalculatorFFT_MPI.hh	

compile : $(OBJS)
	if [ ! -d $(UAL_SIMBAD)/include/SIMBAD/SC ] ; then mkdir $(UAL_SIMBAD)/include/SIMBAD/SC ; fi;
	cp $(HHS) $(UAL_SIMBAD)/include/SIMBAD/SC/.

clean:	
	rm -f $(OBJS) 
	rm -rf $(UAL_SIMBAD)/include/SIMBAD/SC
