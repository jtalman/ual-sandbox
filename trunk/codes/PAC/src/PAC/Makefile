include $(UAL_PAC)/src/Makefile.config

DIRS = ./Common ./Beam

OBJS = 

include ./Common/Objects
include ./Beam/Objects

compile : $(UAL_PAC)/lib/$(UAL_ARCH)/obj/dummy.o
	(if [ !  -d $(UAL_PAC)/include/PAC ]; then mkdir  $(UAL_PAC)/include/PAC; fi;)
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done
	$(LD) $(LDFLAGS) $(INC) -o $(UAL_PAC)/lib/$(UAL_ARCH)/obj/dummy   \
		$(UAL_PAC)/lib/$(UAL_ARCH)/obj/dummy.o $(OBJS) $(LIBS)
	cp /dev/null $(UAL_PAC)/lib/$(UAL_ARCH)/obj/dummy
	$(DLD) $(DLDFLAGS) -o $(UAL_PAC)/lib/$(UAL_ARCH)/libPac.so $(OBJS) $(LIBS)

clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 
	rm -rf $(UAL_PAC)/include/PAC
	rm -f $(UAL_PAC)/lib/$(UAL_ARCH)/obj/dummy.o
	rm -f $(UAL_PAC)/lib/$(UAL_ARCH)/libPac.so

