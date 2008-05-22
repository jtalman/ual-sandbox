include $(UAL)/env/$(UAL_ARCH)/Makefile.config

DIRS = ./Base ./Propagator

OBJS = 

include ./Base/Objects
include ./Propagator/Objects

compile : 
	(if [ !  -d $(UAL_TIBETAN)/include/TIBETAN ]; \
		then mkdir  $(UAL_TIBETAN)/include/TIBETAN; fi;)
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done
	$(DLD) $(DLDFLAGS) -o $(UAL_TIBETAN)/lib/$(UAL_ARCH)/libTibetan.so $(OBJS) $(LIBS)
	cp *.i $(UAL_TIBETAN)/include/TIBETAN/.


clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 
	rm -rf $(UAL_TIBETAN)/include/TIBETAN
	rm -f $(UAL_TIBETAN)/lib/$(UAL_ARCH)/libTibetan.so
