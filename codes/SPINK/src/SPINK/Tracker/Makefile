include $(UAL_SPINK)/src/Makefile.config

include ./Objects

compile : $(OBJS)
	if [ ! -d $(UAL_SPINK)/include/SPINK/Tracker ] ; then mkdir $(UAL_SPINK)/include/SPINK/Tracker ; fi;
	cp *.hh $(UAL_SPINK)/include/SPINK/Tracker/.

clean:	
	rm -f $(OBJS) 
	rm -rf $(UAL_SPINK)/include/SPINK/Tracker
