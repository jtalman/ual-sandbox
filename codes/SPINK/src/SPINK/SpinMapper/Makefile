include $(UAL_SPINK)/src/Makefile.config

INC += -I$(UAL_PAC)/include

include ./Objects

compile : $(OBJS)
	if [ ! -d $(UAL_SPINK)/include/SPINK/SpinMapper ] ; then mkdir $(UAL_SPINK)/include/SPINK/SpinMapper ; fi;
	cp *.hh $(UAL_SPINK)/include/SPINK/SpinMapper/.

clean:	
	rm -f $(OBJS) 
	rm -rf $(UAL_SPINK)/include/SPINK/SpinMapper
