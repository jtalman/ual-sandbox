include $(UAL_PAC)/src/Makefile.config

include ./Objects

compile : $(OBJS)
	if [ ! -d $(UAL_PAC)/include/PAC/Beam ] ; then mkdir $(UAL_PAC)/include/PAC/Beam ; fi;
	cp *.hh $(UAL_PAC)/include/PAC/Beam/.

clean:	
	rm -f $(OBJS) 
	rm -rf $(UAL_PAC)/include/PAC/Beam


