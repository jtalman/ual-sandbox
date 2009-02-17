include $(UAL_SPINK)/src/Makefile.config

INC += -I$(UAL_PAC)/include

include ./Objects

compile : $(OBJS)
	if [ ! -d $(UAL_SPINK)/include/SPINK/Propagator ] ; then mkdir $(UAL_SPINK)/include/SPINK/Propagator ; fi;
	cp *.hh $(UAL_SPINK)/include/SPINK/Propagator/.

clean:	
	rm -f $(OBJS) 
	rm -rf $(UAL_SPINK)/include/SPINK/Propagator
