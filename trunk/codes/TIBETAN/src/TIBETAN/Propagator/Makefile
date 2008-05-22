include $(UAL)/env/$(UAL_ARCH)/Makefile.config

INC +=  -I$(UAL_TIBETAN)/src \
	-I$(UAL_TEAPOT)/src \
	-I$(UAL_PAC)/src \
	-I$(UAL_ZLIB)/src \
	-I$(UAL_CORE)/include

LIBS += -L$(UAL_TIBETAN)/lib/$(UAL_ARCH) \
	-L$(UAL_TEAPOT)/lib/$(UAL_ARCH) \
	-L$(UAL_PAC)/lib/$(UAL_ARCH) \
	-L$(UAL_CORE)/lib/$(UAL_ARCH) \
	-lPac -lUal

include ./Objects

compile : $(OBJS)
	if [ ! -d $(UAL_TIBETAN)/include/TIBETAN/Propagator ] ; \
		then mkdir $(UAL_TIBETAN)/include/TIBETAN/Propagator ; fi;
	cp *.hh $(UAL_TIBETAN)/include/TIBETAN/Propagator/.

clean:	
	rm -f $(OBJS) 
	rm -rf $(UAL_TIBETAN)/include/TIBETAN/Propagator

$(UAL_TIBETAN)/lib/$(UAL_ARCH)/obj/%.o : %.cc
	$(CC) $(CCFLAGS) $(INC) -c $< -o $@;
