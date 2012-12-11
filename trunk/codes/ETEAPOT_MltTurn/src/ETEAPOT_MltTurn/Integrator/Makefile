include $(UAL_ETEAPOT_MltTurn)/src/Makefile.config

INC += -I$(UAL_PAC)/include/
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPac

include ./Objects

compile : $(OBJS)
	if [ ! -d $(UAL_ETEAPOT_MltTurn)/include/ETEAPOT_MltTurn/Integrator ] ; \
		then mkdir $(UAL_ETEAPOT_MltTurn)/include/ETEAPOT_MltTurn/Integrator ; fi;
	cp *.hh $(UAL_ETEAPOT_MltTurn)/include/ETEAPOT_MltTurn/Integrator/.

clean:	
	rm -f $(OBJS) 
	rm -rf $(UAL_ETEAPOT_MltTurn)/include/ETEAPOT_MltTurn/Integrator
