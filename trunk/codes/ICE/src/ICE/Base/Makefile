include $(UAL)/env/$(UAL_ARCH)/Makefile.config

INC +=  -I$(UAL_ICE)/src \
	-I$(UAL_PAC)/src \
	-I$(UAL_ZLIB)/src \
	-I$(UAL_CORE)/include

LIBS += -L$(UAL_ICE)/lib/$(UAL_ARCH) \
	-L$(UAL_PAC)/lib/$(UAL_ARCH) \
	-L$(UAL_CORE)/lib/$(UAL_ARCH) \
	-lPac -lUal

include ./Objects

compile : $(OBJS)
	(if [ !  -d $(UAL_ICE)/include/ICE/Base ]; \
		then mkdir $(UAL_ICE)/include/ICE/Base; fi;)
	cp *.hh $(UAL_ICE)/include/ICE/Base/.	

clean:
	rm -f $(OBJS)
	rm -rf $(UAL_ICE)/include/ICE/Base

$(UAL_ICE)/lib/$(UAL_ARCH)/obj/%.o : %.cc
	$(CC) $(CCFLAGS) $(INC) -c $< -o $@;
