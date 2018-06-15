include $(UAL_EMTEAPOT)/src/Makefile.config

INC += -I$(UAL_PAC)/include/
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPac

include ./Objects

compile : $(OBJS)
	if [ ! -d $(UAL_EMTEAPOT)/include/EMTEAPOT/Integrator ] ; \
		then mkdir $(UAL_EMTEAPOT)/include/EMTEAPOT/Integrator ; fi;
	cp *.hh $(UAL_EMTEAPOT)/include/EMTEAPOT/Integrator/.

clean:	
	rm -f $(OBJS) 
	rm -rf $(UAL_EMTEAPOT)/include/EMTEAPOT/Integrator
