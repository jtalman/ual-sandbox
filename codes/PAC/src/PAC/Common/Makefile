include $(UAL_PAC)/src/Makefile.config

include ./Objects

compile : $(OBJS) 
	if [ ! -d $(UAL_PAC)/include/PAC/Common ] ; then mkdir $(UAL_PAC)/include/PAC/Common ; fi;
	cp *.hh $(UAL_PAC)/include/PAC/Common/.

clean:	
	rm -f $(OBJS) 
	rm -rf $(UAL_PAC)/include/PAC/Common

