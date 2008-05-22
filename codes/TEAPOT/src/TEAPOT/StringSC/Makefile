include $(UAL_TEAPOT)/src/Makefile.config

INC += -I$(UAL_PAC)/include/
INC += -I$(UAL)/tools/include/
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPac
LIBS += -L$(UAL)/tools/lib/$(UAL_ARCH) 		-lgsl -lgslcblas

include ./Objects

compile : $(OBJS)
	if [ ! -d $(UAL_TEAPOT)/include/TEAPOT/StringSC ] ; \
		then mkdir $(UAL_TEAPOT)/include/TEAPOT/StringSC ; fi;
	cp *.hh $(UAL_TEAPOT)/include/TEAPOT/StringSC/.

clean:	
	rm -f $(OBJS) 
	rm -rf $(UAL_TEAPOT)/include/TEAPOT/StringSC
