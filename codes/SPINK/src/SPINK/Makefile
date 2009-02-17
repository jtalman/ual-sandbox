include $(UAL_SPINK)/src/Makefile.config

DIRS = ./Propagator

OBJS = 

include ./Propagator/Objects

compile : 
	(if [ !  -d $(UAL_SPINK)/include/SPINK ]; then mkdir  $(UAL_SPINK)/include/SPINK; fi;)
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done
	$(DLD) $(DLDFLAGS) -o $(UAL_SPINK)/lib/$(UAL_ARCH)/libSpink.so $(OBJS) $(LIBS)

clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 
	rm -rf $(UAL_SPINK)/include/SPINK
	rm -f $(UAL_SPINK)/lib/$(UAL_ARCH)/libSpink.so
