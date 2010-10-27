include $(UAL_ETEAPOT)/src/Makefile.config

INC += -I$(UAL_PAC)/include/
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPac

include ./Objects

compile : $(OBJS)
	if [ ! -d $(UAL_ETEAPOT)/include/ETEAPOT/Integrator ] ; \
		then mkdir $(UAL_ETEAPOT)/include/ETEAPOT/Integrator ; fi;
	cp *.hh *.icc $(UAL_ETEAPOT)/include/ETEAPOT/Integrator/.

clean:	
	rm -f $(OBJS) 
	rm -rf $(UAL_ETEAPOT)/include/ETEAPOT/Integrator
