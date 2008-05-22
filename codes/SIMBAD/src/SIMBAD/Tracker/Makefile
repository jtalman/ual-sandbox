include $(UAL_SIMBAD)/src/Makefile.config

include ./Objects

compile : $(OBJS)
	if [ ! -d $(UAL_SIMBAD)/include/SIMBAD/Tracker ] ; then mkdir $(UAL_SIMBAD)/include/SIMBAD/Tracker ; fi;
	cp *.hh $(UAL_SIMBAD)/include/SIMBAD/Tracker/.

clean:	
	rm -f $(OBJS) 
	rm -rf $(UAL_SIMBAD)/include/SIMBAD/Tracker
