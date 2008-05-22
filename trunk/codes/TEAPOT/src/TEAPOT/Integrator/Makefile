include $(UAL_TEAPOT)/src/Makefile.config

INC += -I$(UAL_PAC)/include/
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPac

include ./Objects

compile : $(OBJS)
	if [ ! -d $(UAL_TEAPOT)/include/TEAPOT/Integrator ] ; \
		then mkdir $(UAL_TEAPOT)/include/TEAPOT/Integrator ; fi;
	cp *.hh *.icc $(UAL_TEAPOT)/include/TEAPOT/Integrator/.

clean:	
	rm -f $(OBJS) 
	rm -rf $(UAL_TEAPOT)/include/TEAPOT/Integrator
