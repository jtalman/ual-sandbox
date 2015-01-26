include $(UAL_ETEAPOT2)/src/Makefile.config

INC += -I$(UAL_PAC)/include/
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPac

include ./Objects

compile : $(OBJS)
	if [ ! -d $(UAL_ETEAPOT2)/include/ETEAPOT2/Integrator ] ; \
		then mkdir $(UAL_ETEAPOT2)/include/ETEAPOT2/Integrator ; fi;
	cp *.hh $(UAL_ETEAPOT2)/include/ETEAPOT2/Integrator/.

clean:	
	rm -f $(OBJS) 
	rm -rf $(UAL_ETEAPOT2)/include/ETEAPOT2/Integrator
